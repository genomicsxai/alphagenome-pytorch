"""Convert an alphagenome_ft (JAX) finetuned checkpoint to a PyTorch delta checkpoint.

An alphagenome_ft checkpoint directory holds ``config.json`` (head configs, optional
``backbone_lora`` and ``head_metadata``) plus an orbax ``checkpoint/`` with only the
custom-head parameters and any q/v LoRA ``lora_a``/``lora_b`` leaves. This module maps
those onto the PyTorch finetuning layout (``TransferConfig`` + head/adapter state dict)
and writes a ``.delta.pth`` that :func:`load_finetuned_model` loads directly, on top of
the matching converted pretrained weights.

Supported: heads-only and q/v backbone-LoRA checkpoints with predefined genome-track heads
(rna_seq, atac, dnase, cage, procap, chip_tf, chip_histone) and the three splice heads.
Everything else (custom heads, other LoRA targets, full/minimal-model or encoder-output
checkpoints) raises a ``ValueError`` naming the unsupported piece.

Per-track nonzero means come from ``head_metadata`` in ``config.json``. Older checkpoints
without it need ``track_means`` (explicit values per head) or ``targets_config`` (the
alphagenome_ft targets YAML/JSON with ``nonzero_mean`` per target).
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from alphagenome_pytorch.jax_compat.transforms import apply_transform
from alphagenome_pytorch.jax_compat.weight_mapping import map_pytorch_to_jax

__all__ = [
    "FinetunedConversion",
    "build_finetuned_conversion",
    "convert_finetuned_checkpoint",
    "is_alphagenome_ft_checkpoint",
    "load_alphagenome_ft_checkpoint",
    "parse_track_means",
]

# JAX OutputType name -> PyTorch genome-track modality.
_TRACK_OUTPUT_TYPES = {
    "ATAC": "atac",
    "CAGE": "cage",
    "DNASE": "dnase",
    "RNA_SEQ": "rna_seq",
    "CHIP_HISTONE": "chip_histone",
    "CHIP_TF": "chip_tf",
    "PROCAP": "procap",
}
# JAX splice head type -> (PyTorch head name, PyTorch module attribute).
_SPLICE_TYPES = {
    "SPLICE_SITES_CLASSIFICATION": ("splice_site", "splice_sites_classification_head"),
    "SPLICE_SITES_USAGE": ("splice_usage", "splice_sites_usage_head"),
    "SPLICE_SITES_JUNCTION": ("splice_junctions", "splice_sites_junction_head"),
}
# Head segment that map_pytorch_to_jax uses for the pretrained splice heads.
_PRETRAINED_SPLICE_IDS = {
    "splice_sites_classification": "splice_site",
    "splice_sites_usage": "splice_usage",
    "splice_sites_junction": "splice_junctions",
}
_SPLICE_CLASS_NAMES = [
    "cls_donor_pos", "cls_acceptor_pos", "cls_donor_neg", "cls_acceptor_neg", "cls_none",
]
# Order of the organism slots in a JAX head built without --organism
# (alphagenome_ft builds metadata for every dna_client.Organism, in enum order).
_ORGANISMS = ("HOMO_SAPIENS", "MUS_MUSCULUS")
_ORGANISM_PT_NAMES = {"HOMO_SAPIENS": "human", "MUS_MUSCULUS": "mouse"}
_ORGANISM_PT_INDEX = {"HOMO_SAPIENS": 0, "MUS_MUSCULUS": 1}
_LORA_TARGETS = {"q_proj": "q_proj", "q_layer": "q_proj", "v_proj": "v_proj", "v_layer": "v_proj"}
_NUM_TOWER_BLOCKS = 9
_JUNCTION_HIDDEN_DIM = 768


@dataclass
class FinetunedConversion:
    """Result of mapping an alphagenome_ft checkpoint onto the PyTorch finetuning layout."""

    transfer_config: Any  # TransferConfig
    state_dict: dict[str, torch.Tensor]  # adapter + head tensors (incl. track_means)
    metadata: dict[str, Any]  # kwargs for save_delta_checkpoint


def is_alphagenome_ft_checkpoint(path: str | Path) -> bool:
    """True for an alphagenome_ft finetuned checkpoint dir (``config.json`` with ``custom_heads``)."""
    config_path = Path(path) / "config.json"
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(config, dict) and "custom_heads" in config


def _flatten(tree: Mapping[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten(value, path))
        else:
            flat[path] = np.asarray(value)
    return flat


def _normalize_param_paths(params: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Flatten a Haiku param dict; head params may be flat ``head/...`` or nested under ``alphagenome``."""
    flat = {}
    for path, value in _flatten(params).items():
        if path.startswith("alphagenome/head/"):
            path = path[len("alphagenome/"):]
        flat[path] = value.astype(np.float32) if value.dtype != np.float32 else value
    return flat


def load_alphagenome_ft_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read ``config.json`` and the orbax params of an alphagenome_ft checkpoint dir."""
    path = Path(path).expanduser().resolve()
    if not is_alphagenome_ft_checkpoint(path):
        raise ValueError(
            f"{path} is not an alphagenome_ft checkpoint directory "
            "(expected config.json with 'custom_heads' and a checkpoint/ subdirectory)."
        )
    config = json.loads((path / "config.json").read_text())
    ckpt_dir = path / "checkpoint"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Missing orbax checkpoint directory: {ckpt_dir}")
    import jax
    import orbax.checkpoint as ocp

    # Restore every leaf as a host numpy array: a target-less restore fails when the
    # checkpoint was written from a different device topology (e.g. saved on GPU).
    metadata = ocp.StandardCheckpointer().metadata(str(ckpt_dir))
    tree = getattr(metadata, "item_metadata", metadata)
    tree = getattr(tree, "tree", tree)
    restore_args = jax.tree_util.tree_map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), tree)
    restored = ocp.PyTreeCheckpointer().restore(
        str(ckpt_dir), args=ocp.args.PyTreeRestore(restore_args=restore_args),
    )
    params = restored[0] if isinstance(restored, (list, tuple)) else restored
    return config, _normalize_param_paths(params)


def parse_track_means(values: Sequence[str] | Mapping[str, Sequence[float]] | None) -> dict[str, list[float]]:
    """Parse ``HEAD=v1,v2,...`` strings (the ``--track-means`` CLI form) into a dict."""
    if values is None:
        return {}
    if isinstance(values, Mapping):
        return {str(k): [float(v) for v in vs] for k, vs in values.items()}
    parsed: dict[str, list[float]] = {}
    for item in values:
        head, sep, rest = item.partition("=")
        if not sep or not head or not rest:
            raise ValueError(f"--track-means expects HEAD=v1,v2,..., got {item!r}")
        try:
            parsed[head.strip()] = [float(v) for v in rest.split(",")]
        except ValueError as exc:
            raise ValueError(f"--track-means {item!r}: values must be numbers") from exc
    return parsed


def _load_targets_config(source: str | Path | Mapping[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    """Per-head target rows (name, strand, nonzero_mean) from an alphagenome_ft targets config."""
    if source is None:
        return {}
    if isinstance(source, Mapping):
        config = source
    else:
        path = Path(source)
        text = path.read_text()
        if path.suffix.lower() in {".yml", ".yaml"}:
            import yaml

            config = yaml.safe_load(text)
        else:
            config = json.loads(text)
    heads: dict[str, list[dict[str, Any]]] = {}
    for entry in (config or {}).get("heads") or []:
        if "id" not in entry or not entry.get("targets"):
            continue
        rows = []
        for target in entry["targets"]:
            if isinstance(target, str):
                target = {"path": target}
            rows.append({
                "name": str(target.get("label") or Path(str(target.get("path", ""))).stem),
                "strand": target.get("strand"),
                "nonzero_mean": target.get("nonzero_mean"),
            })
        heads[str(entry["id"])] = rows
    return heads


def _canonical_organism(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    aliases = {"HUMAN": "HOMO_SAPIENS", "MOUSE": "MUS_MUSCULUS", "0": "HOMO_SAPIENS", "1": "MUS_MUSCULUS"}
    text = aliases.get(text, text)
    if text not in _ORGANISMS:
        raise ValueError(f"Unknown organism {value!r}; expected human, mouse, HOMO_SAPIENS or MUS_MUSCULUS.")
    return text


def _lora_jax_key(pt_name: str) -> str | None:
    match = re.fullmatch(r"tower\.blocks\.(\d+)\.mha\.([qv])_proj\.lora_([AB])\.weight", pt_name)
    if match is None:
        return None
    block, proj, ab = match.groups()
    suffix = "" if block == "0" else f"_{block}"
    return f"alphagenome/transformer_tower/mha_block{suffix}/{proj}_layer/lora_{ab.lower()}"


def _head_jax_key(pt_name: str, jax_ids: Mapping[str, str]) -> str | None:
    """Finetuned JAX path for a PyTorch head parameter (``jax_ids``: PyTorch head name -> JAX id)."""
    pretrained = map_pytorch_to_jax(pt_name)
    if pretrained is None or not pretrained.startswith("alphagenome/head/"):
        return None
    parts = pretrained[len("alphagenome/"):].split("/")
    name = _PRETRAINED_SPLICE_IDS.get(parts[1], parts[1]) if pt_name.startswith("splice_sites_") else parts[1]
    parts[1] = jax_ids[name]
    return "/".join(parts)


def _select_organism(
    head_id: str,
    num_slots: int,
    metadata_organisms: list[str] | None,
    requested: str | None,
) -> tuple[int, str]:
    """Return (slot index, organism enum name) for one head."""
    organisms = metadata_organisms
    if organisms is not None and len(organisms) != num_slots:
        raise ValueError(
            f"Head '{head_id}' has {num_slots} organism slot(s) but head_metadata lists "
            f"{len(organisms)} organism(s) {organisms}; cannot tell which slot is which."
        )
    if organisms is None and num_slots > 1:
        if num_slots > len(_ORGANISMS):
            raise ValueError(f"Head '{head_id}' has {num_slots} organism slots; expected at most 2.")
        organisms = list(_ORGANISMS[:num_slots])
    if num_slots == 1:
        organism = organisms[0] if organisms else (requested or "HOMO_SAPIENS")
        if requested is not None and organism != requested:
            raise ValueError(
                f"Head '{head_id}' was trained for {organism}, but organism {requested} was requested."
            )
        return 0, organism
    if requested is None:
        raise ValueError(
            f"Head '{head_id}' has {num_slots} organism slots {organisms}; "
            "pass an organism (--organism human|mouse) to select one."
        )
    if requested not in organisms:
        raise ValueError(f"Head '{head_id}' has no slot for organism {requested} (has {organisms}).")
    return organisms.index(requested), requested


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def build_finetuned_conversion(
    config: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    organism: str | None = None,
    targets_config: str | Path | Mapping[str, Any] | None = None,
    track_means: Sequence[str] | Mapping[str, Sequence[float]] | None = None,
) -> FinetunedConversion:
    """Map an alphagenome_ft ``config.json`` + param tree onto the PyTorch finetuning layout.

    Args:
        config: Parsed ``config.json``.
        params: Haiku params as saved (flat ``head/...`` keys and/or nested dicts).
        organism: Organism to extract when a head has several organism slots
            (``human``/``mouse`` or the enum name). Defaults to the checkpoint's single organism.
        targets_config: alphagenome_ft targets config (path or parsed mapping) supplying
            per-track ``nonzero_mean`` for checkpoints without ``head_metadata``.
        track_means: Explicit per-head means (``{"head": [..]}`` or ``["head=v1,v2"]``);
            take precedence over ``head_metadata`` and ``targets_config``.
    """
    from alphagenome_pytorch.extensions.finetuning.heads import ASSAY_TYPES, create_finetuning_head
    from alphagenome_pytorch.extensions.finetuning.transfer import TransferConfig

    if config.get("save_full_model") or config.get("save_minimal_model"):
        kind = "full-model" if config.get("save_full_model") else "minimal-model"
        raise ValueError(
            f"Unsupported: {kind} alphagenome_ft checkpoint. Only heads-only checkpoints "
            "(optionally with q/v backbone LoRA; the default save_checkpoint layout) can be converted."
        )
    if config.get("use_encoder_output"):
        raise ValueError("Unsupported: encoder-output (use_encoder_output=True) alphagenome_ft heads.")

    flat = _normalize_param_paths(params)
    requested = _canonical_organism(organism)
    explicit_means = parse_track_means(track_means)
    target_rows = _load_targets_config(targets_config)
    head_metadata = config.get("head_metadata") or {}
    head_configs = config.get("head_configs") or {}

    new_heads: dict[str, dict[str, Any]] = {}
    jax_ids: dict[str, str] = {}  # PyTorch head name -> JAX head id
    module_prefixes: dict[str, str] = {}
    track_names: dict[str, list[str]] = {}
    resolutions: dict[str, tuple[int, ...]] = {}
    track_metadata: list[dict[str, Any]] = []
    slots: dict[str, int] = {}
    organisms: set[str] = set()
    track_means_tensors: dict[str, torch.Tensor] = {}

    for head_id in config["custom_heads"]:
        head_cfg = head_configs.get(head_id)
        if head_cfg is None:
            raise ValueError(f"config.json has no head_configs entry for head '{head_id}'.")
        if head_cfg.get("source") != "predefined":
            raise ValueError(
                f"Unsupported: head '{head_id}' is a custom (source={head_cfg.get('source')!r}) "
                "head; only predefined alphagenome_ft heads can be converted."
            )
        head_type = head_cfg.get("type")
        if head_type == "GENOME_TRACKS":
            output_type = head_cfg.get("output_type")
            if output_type not in _TRACK_OUTPUT_TYPES:
                raise ValueError(f"Unsupported: head '{head_id}' output_type {output_type!r}.")
            modality = _TRACK_OUTPUT_TYPES[output_type]
            pt_name = head_id
            res = tuple(int(r) for r in head_cfg.get("resolutions") or ASSAY_TYPES[modality]["default_resolutions"])
            squash = head_cfg.get("apply_squashing")
            if squash is not None and bool(squash) != ASSAY_TYPES[modality]["apply_squashing"]:
                raise ValueError(
                    f"Unsupported: head '{head_id}' sets apply_squashing={squash}, but PyTorch "
                    f"{modality} heads always use apply_squashing={ASSAY_TYPES[modality]['apply_squashing']}."
                )
            w = flat.get(f"head/{head_id}/resolution_{res[0]}/multi_organism_linear/w")
            if w is None:
                raise ValueError(f"Checkpoint is missing parameters for head '{head_id}' at resolution {res[0]}.")
            num_slots, num_tracks = w.shape[0], w.shape[-1]
            module_prefixes[pt_name] = f"heads.{pt_name}."
        elif head_type in _SPLICE_TYPES:
            modality, attr = _SPLICE_TYPES[head_type]
            pt_name = modality
            if pt_name in new_heads:
                raise ValueError(f"Unsupported: more than one {head_type} head ({jax_ids[pt_name]!r}, {head_id!r}).")
            res = (1,)
            w = flat.get(f"head/{head_id}/multi_organism_linear/w")
            if w is None:
                raise ValueError(f"Checkpoint is missing parameters for head '{head_id}'.")
            num_slots = w.shape[0]
            if modality == "splice_junctions":
                emb = flat[f"head/{head_id}/pos_donor_logits/embeddings"]
                num_tracks = emb.shape[1] // (2 * _JUNCTION_HIDDEN_DIM)
            else:
                num_tracks = w.shape[-1]
            module_prefixes[pt_name] = f"{attr}."
        else:
            raise ValueError(f"Unsupported: head '{head_id}' type {head_type!r}.")

        meta_by_org = head_metadata.get(head_id)
        slot, org = _select_organism(
            head_id, num_slots, list(meta_by_org) if meta_by_org else None, requested,
        )
        rows = meta_by_org[org] if meta_by_org else None
        expected_rows = num_tracks
        if rows is not None and len(rows) != expected_rows:
            raise ValueError(
                f"head_metadata for '{head_id}' ({org}) has {len(rows)} rows, "
                f"but the parameters have {expected_rows} tracks."
            )
        jax_ids[pt_name] = head_id
        slots[pt_name] = slot
        organisms.add(org)
        resolutions[pt_name] = res
        new_heads[pt_name] = {
            "modality": modality,
            "num_tracks": num_tracks,
            "resolutions": list(res),
            "encoder_only": False,
            "num_organisms": 1,
        }

        # Track names, metadata rows and (track heads only) nonzero means.
        if modality in _TRACK_OUTPUT_TYPES.values():
            if rows is None:
                rows = target_rows.get(head_id)
                if rows is not None and len(rows) != num_tracks:
                    raise ValueError(
                        f"targets_config lists {len(rows)} targets for head '{head_id}', "
                        f"but the parameters have {num_tracks} tracks."
                    )
            if head_id in explicit_means:
                means = explicit_means[head_id]
                if len(means) != num_tracks:
                    raise ValueError(
                        f"track_means for head '{head_id}' has {len(means)} values; expected {num_tracks}."
                    )
            elif rows is not None and any(r.get("nonzero_mean") is not None for r in rows):
                # alphagenome_ft defaults unspecified per-track means to 1.0; null stands for NaN.
                means = [
                    (float("nan") if head_metadata.get(head_id) else 1.0)
                    if r.get("nonzero_mean") is None else float(r["nonzero_mean"])
                    for r in rows
                ]
            elif head_metadata.get(head_id):
                warnings.warn(
                    f"head_metadata for '{head_id}' has no nonzero_mean; using 1.0 (no scaling), "
                    "as the JAX head did.",
                    UserWarning,
                    stacklevel=2,
                )
                means = [1.0] * num_tracks
            else:
                raise ValueError(
                    f"No track means for head '{head_id}': config.json has no head_metadata "
                    "(checkpoint saved by an older alphagenome_ft). Pass the training targets "
                    "config with per-target nonzero_mean (--targets-config) or the values "
                    f"directly (--track-means {head_id}=v1,v2,... with {num_tracks} values)."
                )
            track_means_tensors[pt_name] = torch.tensor([means], dtype=torch.float32)
            names = [str(r["name"]) for r in rows] if rows else [f"track_{i}" for i in range(num_tracks)]
            meta_rows = [dict(r) for r in rows] if rows else [{} for _ in names]
        elif modality == "splice_site":
            names = list(_SPLICE_CLASS_NAMES)
            meta_rows = [{} for _ in names]
        elif modality == "splice_usage":
            names = [str(r["name"]) for r in rows] if rows else [f"usage_{i}" for i in range(num_tracks)]
            meta_rows = [dict(r) for r in rows] if rows else [{} for _ in names]
        else:  # splice_junctions: one JAX row per tissue, PyTorch names both strands
            tissues = [str(r["name"]) for r in rows] if rows else [f"tissue_{i}" for i in range(num_tracks)]
            base = [dict(r) for r in rows] if rows else [{} for _ in tissues]
            names = [f"{t}_pos" for t in tissues] + [f"{t}_neg" for t in tissues]
            meta_rows = [{**r, "strand": "+"} for r in base] + [{**r, "strand": "-"} for r in base]
        track_names[pt_name] = names
        # Rows as the finetuning runner records them (TrackMetadataCatalog.to_rows keys).
        for index, (name, row) in enumerate(zip(names, meta_rows)):
            row = {k: _json_safe(v) for k, v in row.items() if k != "name"}
            track_metadata.append({
                **row,
                "output_name": pt_name,
                "track_index": index,
                "track_name": name,
                "organism": _ORGANISM_PT_INDEX[org],
            })

    if len(organisms) > 1:
        raise ValueError(f"Heads were trained for different organisms {sorted(organisms)}; cannot combine.")
    organism_name = _ORGANISM_PT_NAMES[organisms.pop()] if organisms else "human"

    # Backbone LoRA.
    lora_cfg = config.get("backbone_lora")
    lora_paths = sorted(p for p in flat if p.rsplit("/", 1)[-1] in {"lora_a", "lora_b"})
    if lora_cfg is None and lora_paths:
        raise ValueError(f"Checkpoint has LoRA leaves {lora_paths[:2]}... but config.json has no backbone_lora.")
    lora_targets: list[str] = []
    if lora_cfg is not None:
        unknown = sorted(set(lora_cfg.get("targets") or []) - set(_LORA_TARGETS))
        if unknown:
            raise ValueError(f"Unsupported: backbone LoRA targets {unknown}; only q_proj/v_proj can be converted.")
        lora_targets = list(dict.fromkeys(_LORA_TARGETS[t] for t in lora_cfg["targets"]))

    # Expected PyTorch keys/shapes: build each head on its own (cheap) to get its state dict.
    expected: dict[str, tuple[int, ...]] = {}
    for pt_name, head_cfg in new_heads.items():
        head = create_finetuning_head(
            assay_type=head_cfg["modality"],
            n_tracks=head_cfg["num_tracks"],
            resolutions=head_cfg["resolutions"],
            num_organisms=1,
        )
        for key, value in head.state_dict().items():
            expected[module_prefixes[pt_name] + key] = tuple(value.shape)
    rank = int(lora_cfg["rank"]) if lora_cfg is not None else 0
    for block in range(_NUM_TOWER_BLOCKS):
        for target in lora_targets:
            prefix = f"tower.blocks.{block}.mha.{target}"
            b_key = _lora_jax_key(f"{prefix}.lora_B.weight")
            if b_key not in flat:
                raise ValueError(f"Checkpoint is missing LoRA parameter {b_key}.")
            expected[f"{prefix}.lora_A.weight"] = (rank, flat[b_key.replace("lora_b", "lora_a")].shape[0])
            expected[f"{prefix}.lora_B.weight"] = (flat[b_key].shape[1], rank)

    state_dict: dict[str, torch.Tensor] = {}
    used: set[str] = set()
    for pt_key, pt_shape in expected.items():
        if pt_key.endswith(".track_means"):
            state_dict[pt_key] = track_means_tensors[pt_key.split(".")[1]]
            continue
        jax_key = _lora_jax_key(pt_key) or _head_jax_key(pt_key, jax_ids)
        if jax_key is None or jax_key not in flat:
            raise ValueError(f"Checkpoint is missing {jax_key or pt_key} (for PyTorch {pt_key}).")
        array = flat[jax_key]
        if jax_key.startswith("head/"):
            head_name = next(n for n, p in module_prefixes.items() if pt_key.startswith(p))
            slot = slots[head_name]
            array = array[slot:slot + 1]
        state_dict[pt_key] = torch.from_numpy(
            np.ascontiguousarray(apply_transform(pt_key, array, pt_shape))
        )
        used.add(jax_key)

    leftover = sorted(set(flat) - used)
    if leftover:
        offending = [p for p in leftover if p.rsplit("/", 1)[-1] in {"lora_a", "lora_b"}]
        if offending:
            raise ValueError(f"Unsupported: LoRA on targets other than q/v ({offending[:3]}).")
        raise ValueError(
            f"Unsupported: checkpoint parameters outside the converted heads/LoRA: {leftover[:5]}"
            f"{' ...' if len(leftover) > 5 else ''}"
        )

    if lora_cfg is not None:
        alpha = float(lora_cfg.get("alpha", 16))
        transfer_config = TransferConfig(
            mode="lora",
            lora_rank=rank,
            lora_alpha=int(alpha) if alpha.is_integer() else alpha,
            lora_targets=lora_targets,
            new_heads=new_heads,
        )
    else:
        transfer_config = TransferConfig(mode="linear", new_heads=new_heads)

    organism_index = 1 if organism_name == "mouse" else 0
    metadata = {
        "track_names": track_names,
        "modality": list(new_heads),
        "resolutions": resolutions,
        "track_metadata": track_metadata,
        "organism": organism_name,
        "organism_indices": [organism_index],
        "source_format": "alphagenome_ft",
    }
    return FinetunedConversion(transfer_config, state_dict, metadata)


def convert_finetuned_checkpoint(
    checkpoint_dir: str | Path,
    output_path: str | Path,
    *,
    organism: str | None = None,
    targets_config: str | Path | Mapping[str, Any] | None = None,
    track_means: Sequence[str] | Mapping[str, Sequence[float]] | None = None,
    base_weights: str | Path | None = None,
) -> FinetunedConversion:
    """Convert an alphagenome_ft checkpoint dir to a PyTorch ``.delta.pth``.

    The delta is written with :func:`save_delta_checkpoint` on a model prepared with the
    converted :class:`TransferConfig`, so :func:`load_finetuned_model` loads it with the
    converted pretrained weights of the same fold. ``base_weights`` (optional) records
    that file's exact weights hash in the delta. See :func:`build_finetuned_conversion`
    for the remaining arguments.
    """
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.extensions.finetuning.checkpointing import (
        compute_base_model_weights_hash_from_file,
        save_delta_checkpoint,
    )
    from alphagenome_pytorch.extensions.finetuning.transfer import prepare_for_transfer, remove_all_heads

    config, params = load_alphagenome_ft_checkpoint(checkpoint_dir)
    conversion = build_finetuned_conversion(
        config, params, organism=organism, targets_config=targets_config, track_means=track_means,
    )
    model = remove_all_heads(AlphaGenome())
    model = prepare_for_transfer(model, conversion.transfer_config)
    result = model.load_state_dict(conversion.state_dict, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Converted keys not found in the PyTorch model: {result.unexpected_keys[:5]}")
    weights_hash = (
        compute_base_model_weights_hash_from_file(base_weights) if base_weights is not None else None
    )
    save_delta_checkpoint(
        output_path,
        model,
        conversion.transfer_config,
        base_model_weights_hash=weights_hash,
        **conversion.metadata,
    )
    return conversion

"""Shared evaluation primitives for fine-tuned AlphaGenome models.

Consumed by scripts/evaluate_finetuned.py (per-model run), scripts/evaluate_matrix.py
(cross-model × cross-dataset matrix), and scripts/predict_locus.py (manuscript figure).
Single inference + metrics code path keeps the three entry points consistent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.extensions.finetuning.checkpointing import (
    is_delta_checkpoint,
    load_delta_checkpoint,
)
from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head
from alphagenome_pytorch.extensions.finetuning.training import NUM_SEGMENTS
from alphagenome_pytorch.extensions.finetuning.transfer import (
    TransferConfig,
    add_head,
    prepare_for_transfer,
    remove_all_heads,
    transfer_config_from_dict,
)
from alphagenome_pytorch.losses import multinomial_loss
from alphagenome_pytorch.named_outputs import TrackMetadataCatalog

log = logging.getLogger(__name__)


# =============================================================================
# Model loading
# =============================================================================


def _normalize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Pick out the fields the eval path actually uses, coerce types."""
    modality = meta.get("modality")
    if isinstance(modality, list) and len(modality) == 1:
        modality = modality[0]

    resolutions = meta.get("resolutions")
    if isinstance(resolutions, dict):
        if isinstance(modality, str) and modality in resolutions:
            resolutions = tuple(resolutions[modality])
        else:
            first = next(iter(resolutions.values()))
            resolutions = tuple(first)
    elif resolutions is not None:
        resolutions = tuple(resolutions)

    track_names = meta.get("track_names")
    if isinstance(track_names, dict) and isinstance(modality, str):
        track_names = track_names.get(modality, [])

    return {
        "modality": modality,
        "resolutions": resolutions,
        "track_names": track_names,
        "epoch": meta.get("epoch"),
        "val_loss": meta.get("val_loss"),
    }


def _apply_transfer_config_from_json(
    model: nn.Module, config_path: str | Path,
) -> nn.Module:
    with open(config_path) as f:
        cfg_dict = json.load(f)
    config = transfer_config_from_dict(cfg_dict)
    return prepare_for_transfer(model, config)


def _apply_heads_from_metadata(
    model: nn.Module, modality: str, track_names: list[str], resolutions: tuple[int, ...],
) -> nn.Module:
    """Attach a fresh finetuning head matching a full-checkpoint's head shapes.

    For linear-probe / full finetuning, no adapters are needed — only the new
    head must exist before ``load_state_dict``. We remove the pretrained heads
    and add the finetuning head named after ``modality``.
    """
    model = remove_all_heads(model)
    head = create_finetuning_head(
        modality=modality,
        n_tracks=len(track_names),
        resolutions=resolutions,
    )
    add_head(model, modality, head)
    return model


def load_finetuned_model(
    checkpoint_path: str | Path,
    pretrained_weights: str | Path,
    device: torch.device,
    transfer_config_json: str | Path | None = None,
) -> tuple[nn.Module, dict]:
    """Load a finetuned model regardless of checkpoint format.

    Supports:
    - ``best_model.delta.pth`` (delta; TransferConfig embedded).
    - ``best_model.pth`` (full) with adapter modes, if ``transfer_config_json``
      is supplied (dumped alongside training).
    - ``best_model.pth`` (full) without adapters (linear-probe / full mode)
      — architecture is reconstructed from metadata.

    Returns (model, normalized_metadata_dict) with keys: modality, resolutions,
    track_names, epoch, val_loss. Parameters are frozen.
    """
    checkpoint_path = Path(checkpoint_path)
    log.info("Loading checkpoint: %s", checkpoint_path)

    model = AlphaGenome.from_pretrained(str(pretrained_weights), device=device)

    if is_delta_checkpoint(checkpoint_path):
        config, metadata = load_delta_checkpoint(
            checkpoint_path, model, verify_hash=False,
        )
        model.to(device)
    else:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = {
            k: raw[k] for k in
            ("modality", "resolutions", "track_names", "epoch", "val_loss")
            if k in raw
        }
        norm = _normalize_metadata(metadata)

        if transfer_config_json is not None:
            model = _apply_transfer_config_from_json(model, transfer_config_json)
        else:
            has_adapter_keys = any(
                ".lora_" in k or ".locon_" in k or ".scale" in k
                or ".adapter." in k
                for k in raw["model_state_dict"].keys()
            )
            if has_adapter_keys:
                raise RuntimeError(
                    f"Full checkpoint {checkpoint_path.name} contains adapter "
                    "weights but no transfer_config.json was supplied. Pass "
                    "--transfer-config (a JSON dumped from TransferConfig) or "
                    "re-train with --save-delta for a self-describing checkpoint."
                )
            model = _apply_heads_from_metadata(
                model, norm["modality"], norm["track_names"], norm["resolutions"],
            )

        model.load_state_dict(raw["model_state_dict"], strict=False)
        model.to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, _normalize_metadata(metadata)


def load_native_model(
    pretrained_weights: str | Path,
    native_biosample: str | None,
    native_track_index: int | None,
    modality: str,
    device: torch.device,
) -> tuple[nn.Module, int, str]:
    """Load pretrained model with native heads and resolve the comparison track.

    Pass either ``native_biosample`` (substring match, case-insensitive) or
    ``native_track_index``. Returns (model, track_index, display_name).
    """
    model = AlphaGenome.from_pretrained(str(pretrained_weights), device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    catalog = TrackMetadataCatalog.load_builtin("human")
    model.set_track_metadata_catalog(catalog)

    tracks = catalog.get_tracks(modality, organism=0)

    if native_track_index is not None:
        if native_track_index >= len(tracks):
            raise ValueError(
                f"Track index {native_track_index} out of range for "
                f"{modality} ({len(tracks)} tracks)"
            )
        track = tracks[native_track_index]
        display_name = track.get("biosample_name") or track.track_name
        return model, native_track_index, display_name

    query = native_biosample.lower()
    matches = [
        t for t in tracks
        if not t.is_padding and query in (t.get("biosample_name") or "").lower()
    ]
    if not matches:
        available = sorted({
            t.get("biosample_name")
            for t in tracks
            if not t.is_padding and t.get("biosample_name")
        })
        raise ValueError(
            f"No {modality} track found matching biosample '{native_biosample}'. "
            f"Available biosamples ({len(available)}): {available[:20]}"
        )
    if len(matches) > 1:
        log.warning(
            "Multiple tracks match '%s': %s. Using first.",
            native_biosample,
            [(m.track_index, m.get("biosample_name")) for m in matches[:5]],
        )

    track = matches[0]
    display_name = track.get("biosample_name") or track.track_name
    return model, track.track_index, display_name


# =============================================================================
# Inference
# =============================================================================


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    modality: str,
    loader: DataLoader,
    device: torch.device,
    resolutions: tuple[int, ...],
    positional_weight: float = 5.0,
    progress_desc: str = "Evaluating (finetuned)",
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], float]:
    """Run finetuned model inference.

    Returns (preds, targets, avg_loss). Predictions are in experimental
    (unscaled) space so they can be compared directly to BigWig values.
    """
    model.eval()
    head = model.heads[modality]

    preds_by_res: dict[int, list[np.ndarray]] = {r: [] for r in resolutions}
    targets_by_res: dict[int, list[np.ndarray]] = {r: [] for r in resolutions}
    total_loss = 0.0
    n_batches = 0

    for sequences, targets_dict in tqdm(loader, desc=progress_desc):
        sequences = sequences.to(device)
        organism_idx = torch.zeros(
            sequences.shape[0], dtype=torch.long, device=device,
        )

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs = model(
                sequences, organism_idx,
                embeddings_only=True, resolutions=resolutions,
                channels_last=False,
            )
            embeddings_dict = {
                res: outputs[f"embeddings_{res}bp"]
                for res in resolutions
                if f"embeddings_{res}bp" in outputs
            }
            scaled_preds = head(
                embeddings_dict, organism_idx,
                return_scaled=True, channels_last=True,
            )
            exp_preds = head(
                embeddings_dict, organism_idx,
                return_scaled=False, channels_last=True,
            )

        loss = torch.tensor(0.0, device=device)
        for res in resolutions:
            if res not in scaled_preds or res not in targets_dict:
                continue
            pred = scaled_preds[res]
            targets = targets_dict[res].to(device)
            targets_scaled = head.scale(targets, organism_idx, resolution=res)
            mask = torch.ones(
                pred.shape[0], 1, pred.shape[-1],
                dtype=torch.bool, device=device,
            )
            seq_len = pred.shape[-2]
            mn_res = max(1, seq_len // NUM_SEGMENTS)
            while mn_res > 1 and seq_len % mn_res != 0:
                mn_res -= 1
            ld = multinomial_loss(
                y_pred=pred, y_true=targets_scaled, mask=mask,
                multinomial_resolution=mn_res,
                positional_weight=positional_weight,
            )
            loss = loss + ld["loss"]

        total_loss += loss.item()
        n_batches += 1

        for res in resolutions:
            if res in exp_preds:
                preds_by_res[res].append(exp_preds[res].float().cpu().numpy())
            if res in targets_dict:
                targets_by_res[res].append(targets_dict[res].numpy())

    all_preds = {r: np.concatenate(v, axis=0) for r, v in preds_by_res.items() if v}
    all_targets = {
        r: np.concatenate(v, axis=0) for r, v in targets_by_res.items() if v
    }
    avg_loss = total_loss / max(1, n_batches)
    return all_preds, all_targets, avg_loss


@torch.no_grad()
def evaluate_native_split(
    model: nn.Module,
    modality: str,
    track_index: int,
    loader: DataLoader,
    device: torch.device,
    resolutions: tuple[int, ...],
    progress_desc: str = "Evaluating (native)",
) -> dict[int, np.ndarray]:
    """Run native model on the same loader, extract a single track.

    Returns dict[resolution -> (N, seq_len, 1)].
    """
    model.eval()
    preds_by_res: dict[int, list[np.ndarray]] = {r: [] for r in resolutions}

    for sequences, _ in tqdm(loader, desc=progress_desc):
        sequences = sequences.to(device)
        organism_idx = torch.zeros(
            sequences.shape[0], dtype=torch.long, device=device,
        )

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs = model(
                sequences, organism_idx,
                heads=(modality,), resolutions=resolutions,
            )

        head_out = outputs.get(modality)
        if head_out is None:
            continue
        for res in resolutions:
            if res not in head_out:
                continue
            pred = head_out[res]  # (B, S, T) NLC
            pred_track = pred[:, :, track_index : track_index + 1]
            preds_by_res[res].append(pred_track.float().cpu().numpy())

    return {r: np.concatenate(v, axis=0) for r, v in preds_by_res.items() if v}


# =============================================================================
# Metrics
# =============================================================================


def jsd_per_region(
    preds: np.ndarray, targets: np.ndarray, eps: float = 1e-8,
) -> np.ndarray:
    """Jensen-Shannon divergence per region between normalized profiles.

    preds, targets: (N, seq_len, n_tracks). Returns (N, n_tracks).
    """
    p = targets / (targets.sum(axis=1, keepdims=True) + eps)
    q = preds / (preds.sum(axis=1, keepdims=True) + eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)), axis=1)
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)), axis=1)
    return 0.5 * (kl_pm + kl_qm)


def compute_all_metrics(
    preds: np.ndarray, targets: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Compute evaluation metrics from aligned (preds, targets).

    preds, targets: (N, seq_len, n_tracks). Returns a dict with per-region
    arrays (profile_pearson_r_all, jsd_all) and scalar summaries.
    """
    n_regions = preds.shape[0]

    profile_rs = []
    for i in range(n_regions):
        p = preds[i].flatten()
        t = targets[i].flatten()
        if np.std(t) > 1e-10 and np.std(p) > 1e-10:
            r, _ = stats.pearsonr(p, t)
            profile_rs.append(r)
        else:
            profile_rs.append(0.0)
    profile_rs = np.array(profile_rs)

    pred_counts = preds.sum(axis=1).flatten()
    target_counts = targets.sum(axis=1).flatten()
    if np.std(pred_counts) > 1e-10 and np.std(target_counts) > 1e-10:
        count_r = stats.pearsonr(pred_counts, target_counts)[0]
    else:
        count_r = 0.0

    jsd_vals = jsd_per_region(preds, targets)
    jsd_per_reg = jsd_vals.mean(axis=1)

    p_flat = preds.flatten()
    t_flat = targets.flatten()
    if len(p_flat) > 2_000_000:
        idx = np.random.default_rng(42).choice(
            len(p_flat), 2_000_000, replace=False,
        )
        p_flat = p_flat[idx]
        t_flat = t_flat[idx]

    spearman_global = stats.spearmanr(p_flat, t_flat)[0]
    mse = float(np.mean((preds - targets) ** 2))

    return {
        "profile_pearson_r_all": profile_rs,
        "profile_pearson_r_mean": float(np.mean(profile_rs)),
        "profile_pearson_r_median": float(np.median(profile_rs)),
        "count_pearson_r": float(count_r),
        "jsd_all": jsd_per_reg,
        "jsd_mean": float(np.mean(jsd_per_reg)),
        "jsd_median": float(np.median(jsd_per_reg)),
        "mse": mse,
        "spearman_global": float(spearman_global),
        "n_regions": n_regions,
    }


# =============================================================================
# Paired statistics
# =============================================================================


def paired_stats(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired comparison between two per-region metric vectors.

    Pairing: a[i] and b[i] are from the same region. Returns wilcoxon_p
    (two-sided signed-rank), median_diff (median(a-b)), ci_low, ci_high
    (95% bootstrap CI of median diff), and n_pairs.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    diff = a - b
    n = len(diff)

    if n == 0 or np.all(diff == 0):
        return {
            "wilcoxon_p": float("nan"),
            "median_diff": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n_pairs": n,
        }

    try:
        _, p = stats.wilcoxon(a, b, zero_method="wilcox")
    except ValueError:
        p = float("nan")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_medians = np.median(diff[idx], axis=1)
    ci_low, ci_high = np.quantile(boot_medians, [0.025, 0.975])

    return {
        "wilcoxon_p": float(p),
        "median_diff": float(np.median(diff)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_pairs": n,
    }

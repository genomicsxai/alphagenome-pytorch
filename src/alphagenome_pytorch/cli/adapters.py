"""agt adapters — manage exported finetuned adapter bundles."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from alphagenome_pytorch.cli._output import emit_error, emit_json, emit_text


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "adapters",
        help="Manage exported finetuned adapter bundles (export / inspect / validate)",
        description=(
            "Adapter bundle workflow: export a finetuned checkpoint into a "
            "shareable bundle directory, inspect bundle metadata, and "
            "validate compatibility against a base model."
        ),
    )
    sub = p.add_subparsers(dest="adapters_command")

    _register_export(sub)
    _register_inspect(sub)
    _register_validate(sub)
    _register_pull(sub)
    _register_publish(sub)


def _register_export(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "export",
        help="Package a delta checkpoint or delta-weights export into a bundle directory",
    )
    p.add_argument("--checkpoint", required=True,
                   help="Source delta checkpoint (.delta.pth) or delta-weights export (.safetensors).")
    p.add_argument("--out", required=True,
                   help="Output bundle directory (created if missing).")
    p.add_argument("--id", dest="bundle_id", required=True,
                   help="Bundle identifier (slug).")
    p.add_argument("--label", default=None,
                   help="Human-readable label.")
    p.add_argument("--base-model", dest="base_model_id", default=None,
                   help="Identifier of the base model (e.g. an HF repo).")
    p.add_argument("--base-weights", default=None,
                   help="Base weights file. Required for safetensors-source exports "
                        "to compute base_model_hash; optional for .delta.pth sources "
                        "(hash is read from the checkpoint).")
    p.add_argument("--base-model-hash", default=None,
                   help="Override base model hash explicitly (skips loading base weights).")
    p.add_argument("--genome", default=None)
    p.add_argument("--organism", default=None,
                   choices=["human", "mouse"])
    p.add_argument("--modality", default=None)
    p.add_argument("--biosample", default=None)
    p.add_argument("--heads", default=None,
                   help="Comma-separated head names trained in this adapter.")
    p.add_argument("--license", dest="license_name", default=None)
    p.add_argument("--no-readme", action="store_true",
                   help="Do not generate README.md model card.")
    p.add_argument("--metrics", default=None,
                   help="Optional path to a metrics JSON file to copy into the bundle.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing bundle directory.")


def _register_inspect(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "inspect",
        help="Print a bundle's manifest in human or JSON form.",
    )
    p.add_argument("bundle_dir", help="Path to a local bundle directory.")


def _register_validate(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "validate",
        help="Validate a bundle's manifest and (optionally) base-model compatibility.",
    )
    p.add_argument("bundle_dir", help="Path to a local bundle directory.")
    p.add_argument("--base-weights", default=None,
                   help="Base weights file to verify base_model_hash against.")


def _register_pull(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "pull",
        help="Resolve a bundle URI (local or hf://) and print its local path.",
    )
    p.add_argument("uri", help="Bundle URI: bare path, local:..., file://..., or hf://org/repo[/subdir][@revision].")
    p.add_argument("--cache-dir", default=None,
                   help="Override the Hugging Face cache directory (hf:// only).")
    p.add_argument("--token", default=None,
                   help="Hugging Face access token (hf:// only).")


def _register_publish(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "publish",
        help="Upload a local bundle directory to a Hugging Face repository.",
    )
    p.add_argument("bundle_dir", help="Path to a local bundle directory.")
    p.add_argument("repo_uri", help="Target hf://org/repo[/subdir][@revision] URI.")
    p.add_argument("--private", action="store_true",
                   help="Create the repository as private if it does not exist.")
    p.add_argument("--token", default=None,
                   help="Hugging Face access token.")
    p.add_argument("--message", default=None,
                   help="Commit message for the upload.")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    cmd = getattr(args, "adapters_command", None)
    if cmd is None:
        # No subcommand: print top-level help via SystemExit code.
        print("Usage: agt adapters {export,inspect,validate} ...", file=sys.stderr)
        return 1
    if cmd == "export":
        return _run_export(args)
    if cmd == "inspect":
        return _run_inspect(args)
    if cmd == "validate":
        return _run_validate(args)
    if cmd == "pull":
        return _run_pull(args)
    if cmd == "publish":
        return _run_publish(args)
    print(f"Unknown adapters subcommand: {cmd}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def _run_export(args: argparse.Namespace) -> int:
    from alphagenome_pytorch import __version__ as ag_version
    from alphagenome_pytorch.extensions.serving.bundle import (
        DEFAULT_ADAPTER_FILENAME,
        METRICS_FILENAME,
        README_FILENAME,
        Manifest,
        render_model_card,
    )

    src = Path(args.checkpoint)
    if not src.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {src}")

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.force:
            raise FileExistsError(
                f"Output directory {out_dir} is not empty. Pass --force to overwrite."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_out = out_dir / DEFAULT_ADAPTER_FILENAME

    summary = _materialize_adapter_safetensors(src, adapter_out)

    base_hash = _resolve_base_model_hash(args, summary)

    heads_list: list[str] = []
    if args.heads:
        heads_list = [h.strip() for h in args.heads.split(",") if h.strip()]
    elif summary.get("new_head_names"):
        heads_list = list(summary["new_head_names"])

    manifest = Manifest(
        id=args.bundle_id,
        base_model_hash=base_hash,
        label=args.label,
        base_model_id=args.base_model_id,
        alphagenome_pytorch_version=summary.get("alphagenome_pytorch_version") or ag_version,
        adapter_summary=summary.get("adapter_summary", {}),
        genome=args.genome,
        organism=args.organism,
        modality=args.modality,
        biosample=args.biosample,
        heads=heads_list,
        metrics_path=METRICS_FILENAME if args.metrics else None,
        license=args.license_name,
        provenance=summary.get("provenance", {}),
    )
    manifest.dump(out_dir)

    if args.metrics:
        metrics_src = Path(args.metrics)
        if not metrics_src.is_file():
            raise FileNotFoundError(f"Metrics file not found: {metrics_src}")
        shutil.copyfile(metrics_src, out_dir / METRICS_FILENAME)

    if not args.no_readme:
        (out_dir / README_FILENAME).write_text(render_model_card(manifest))

    json_mode = getattr(args, "json_output", False)
    if json_mode:
        emit_json({
            "bundle_dir": str(out_dir),
            "manifest": manifest.to_dict(),
        })
    else:
        emit_text(
            f"Wrote bundle to {out_dir}\n"
            f"  manifest: alphagenome_adapter.json\n"
            f"  weights:  {adapter_out.name}\n"
            f"  id:       {manifest.id}\n"
            f"  base hash: {manifest.base_model_hash}"
        )
    return 0


def _materialize_adapter_safetensors(
    src: Path, dest: Path
) -> dict[str, Any]:
    """Produce ``dest`` (a delta-weights safetensors) from ``src`` and return summary.

    The summary dict captures fields that we want to surface in the manifest:
    ``adapter_summary``, ``new_head_names``, ``alphagenome_pytorch_version``,
    ``provenance``, plus ``base_model_hash`` if the source carried one.
    """
    from alphagenome_pytorch.extensions.finetuning.checkpointing import (
        _read_delta_export_header,
        is_delta_checkpoint,
        is_delta_weights_export,
    )

    if is_delta_weights_export(src):
        # Already in the right format — just copy bytes.
        if src.resolve() != dest.resolve():
            shutil.copyfile(src, dest)
        header = _read_delta_export_header(dest)
        return _summary_from_header(header)

    if is_delta_checkpoint(src):
        return _convert_delta_checkpoint_to_export(src, dest)

    raise ValueError(
        f"Source {src} is neither a delta checkpoint nor a delta-weights export. "
        "Pass a .delta.pth or an exported .safetensors produced by export_delta_weights."
    )


def _convert_delta_checkpoint_to_export(src: Path, dest: Path) -> dict[str, Any]:
    """Re-emit a `.delta.pth` checkpoint as a delta-weights `.safetensors`.

    Reads adapter/head/norm state dicts directly from the checkpoint (no
    model construction needed) and writes them out with the same metadata
    fields ``export_delta_weights`` uses (``transfer_config``, optional
    ``track_names``, optional ``track_metadata``).
    """
    import torch

    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError(
            "safetensors not installed. Install with: pip install safetensors"
        ) from exc

    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{src} is not a recognized delta checkpoint dict")

    weights: dict[str, torch.Tensor] = {}
    for section_key in ("adapter_state_dict", "head_state_dict", "norm_state_dict"):
        section = checkpoint.get(section_key) or {}
        if not isinstance(section, dict):
            raise ValueError(f"{src}: {section_key} is not a dict")
        weights.update(section)
    if not weights:
        raise ValueError(
            f"{src} does not contain adapter/head/norm weights — nothing to export"
        )

    transfer_config = checkpoint.get("transfer_config")
    if transfer_config is None:
        raise ValueError(f"{src} is missing transfer_config; cannot export as bundle")

    metadata: dict[str, str] = {"transfer_config": json.dumps(transfer_config)}

    ckpt_meta = checkpoint.get("metadata") or {}
    if "track_names" in ckpt_meta:
        metadata["track_names"] = json.dumps(ckpt_meta["track_names"])
    if "track_metadata" in ckpt_meta:
        metadata["track_metadata"] = json.dumps(ckpt_meta["track_metadata"])

    save_file(
        {k: v.contiguous().cpu() for k, v in weights.items()},
        str(dest),
        metadata=metadata,
    )

    summary = _summary_from_header({"transfer_config": transfer_config})
    summary["base_model_hash_from_source"] = checkpoint.get("base_model_hash")
    summary["alphagenome_pytorch_version"] = ckpt_meta.get(
        "alphagenome_pytorch_version"
    )
    provenance: dict[str, Any] = {}
    for key in ("created_at", "epoch", "val_loss", "training_run_id", "git_commit"):
        if key in ckpt_meta:
            provenance[key] = ckpt_meta[key]
    summary["provenance"] = provenance
    return summary


def _summary_from_header(header: dict[str, Any]) -> dict[str, Any]:
    """Distil display-worthy fields from a delta-export header."""
    transfer_config = header.get("transfer_config") or {}
    mode = transfer_config.get("mode")
    if isinstance(mode, list):
        kind = mode[0] if mode else None
    else:
        kind = mode
    adapter_summary: dict[str, Any] = {}
    if kind:
        adapter_summary["kind"] = kind
    for key in ("lora_rank", "lora_alpha", "ia3_init", "houlsby_dim"):
        if key in transfer_config:
            adapter_summary[key] = transfer_config[key]
    new_heads = transfer_config.get("new_heads") or {}
    return {
        "adapter_summary": adapter_summary,
        "new_head_names": list(new_heads.keys()) if isinstance(new_heads, dict) else [],
    }


def _resolve_base_model_hash(
    args: argparse.Namespace, summary: dict[str, Any]
) -> str:
    """Determine the base_model_hash to record in the manifest.

    Precedence: explicit ``--base-model-hash`` > computed from
    ``--base-weights`` > value carried in the source `.delta.pth`.
    """
    if args.base_model_hash:
        return args.base_model_hash
    if args.base_weights:
        return _hash_from_base_weights(Path(args.base_weights))
    src_hash = summary.get("base_model_hash_from_source")
    if src_hash:
        return src_hash
    raise ValueError(
        "Cannot determine base_model_hash. Provide --base-weights, "
        "--base-model-hash, or use a .delta.pth source that carries one."
    )


def _hash_from_base_weights(weights_path: Path) -> str:
    """Compute base_model_hash by loading base weights into an AlphaGenome model."""
    import torch
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.extensions.finetuning.checkpointing import (
        compute_base_model_hash,
    )

    if not weights_path.is_file():
        raise FileNotFoundError(f"Base weights file not found: {weights_path}")

    if weights_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(str(weights_path))
    else:
        state_dict = torch.load(
            str(weights_path), map_location="cpu", weights_only=True
        )

    model = AlphaGenome(num_organisms=2)
    # strict=False to tolerate any extra keys (e.g. track_means stored
    # alongside trunk weights). Hashing only inspects trunk keys/shapes.
    model.load_state_dict(state_dict, strict=False)
    return compute_base_model_hash(model)


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def _run_inspect(args: argparse.Namespace) -> int:
    from alphagenome_pytorch.extensions.serving.bundle import (
        BundlePaths,
        Manifest,
    )

    paths = BundlePaths.resolve(args.bundle_dir)
    manifest = Manifest.load(paths.manifest)

    json_mode = getattr(args, "json_output", False)
    if json_mode:
        emit_json({
            "bundle_dir": str(paths.bundle_dir),
            "manifest": manifest.to_dict(),
            "files": {
                "adapter": paths.adapter_safetensors.name,
                "readme": paths.readme.name if paths.readme else None,
                "metrics": paths.metrics.name if paths.metrics else None,
            },
        })
        return 0

    lines = [
        f"Bundle: {paths.bundle_dir}",
        f"  id:           {manifest.id}",
        f"  label:        {manifest.label or '—'}",
        f"  base_model:   {manifest.base_model_id or '—'}",
        f"  base hash:    {manifest.base_model_hash}",
        f"  ag version:   {manifest.alphagenome_pytorch_version or '—'}",
        f"  genome:       {manifest.genome or '—'}",
        f"  organism:     {manifest.organism or '—'}",
        f"  modality:     {manifest.modality or '—'}",
        f"  biosample:    {manifest.biosample or '—'}",
        f"  license:      {manifest.license or '—'}",
        f"  heads:        {', '.join(manifest.heads) if manifest.heads else '—'}",
        f"  adapter:      {paths.adapter_safetensors.name}",
    ]
    if manifest.adapter_summary:
        lines.append("  adapter info:")
        for k, v in sorted(manifest.adapter_summary.items()):
            lines.append(f"    {k}: {v}")
    if manifest.provenance:
        lines.append("  provenance:")
        for k, v in sorted(manifest.provenance.items()):
            lines.append(f"    {k}: {v}")
    emit_text("\n".join(lines))
    return 0


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _run_validate(args: argparse.Namespace) -> int:
    from alphagenome_pytorch.extensions.serving.bundle import validate_bundle

    base_model = None
    if args.base_weights:
        base_model = _build_base_model(Path(args.base_weights))

    report = validate_bundle(args.bundle_dir, base_model=base_model)

    json_mode = getattr(args, "json_output", False)
    if json_mode:
        emit_json(report.to_dict())
    else:
        if report.ok:
            emit_text(f"OK — {report.bundle_dir}")
        else:
            emit_text(f"INVALID — {report.bundle_dir}")
        for w in report.warnings:
            emit_text(f"  warning: {w}")
        for e in report.errors:
            emit_text(f"  error:   {e}")
    return 0 if report.ok else 1


def _build_base_model(weights_path: Path) -> Any:
    import torch
    from alphagenome_pytorch import AlphaGenome

    if not weights_path.is_file():
        raise FileNotFoundError(f"Base weights file not found: {weights_path}")

    if weights_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(str(weights_path))
    else:
        state_dict = torch.load(
            str(weights_path), map_location="cpu", weights_only=True
        )

    model = AlphaGenome(num_organisms=2)
    model.load_state_dict(state_dict, strict=False)
    return model


# ---------------------------------------------------------------------------
# pull / publish
# ---------------------------------------------------------------------------


def _run_pull(args: argparse.Namespace) -> int:
    from alphagenome_pytorch.extensions.serving.uri import (
        parse_bundle_uri,
        resolve_bundle,
    )

    parsed = parse_bundle_uri(args.uri)
    paths = resolve_bundle(
        parsed, cache_dir=args.cache_dir, token=args.token,
    )

    json_mode = getattr(args, "json_output", False)
    if json_mode:
        emit_json({
            "uri": parsed.raw,
            "scheme": parsed.scheme,
            "bundle_dir": str(paths.bundle_dir),
            "manifest": str(paths.manifest),
            "adapter": str(paths.adapter_safetensors),
        })
    else:
        emit_text(str(paths.bundle_dir))
    return 0


def _run_publish(args: argparse.Namespace) -> int:
    from alphagenome_pytorch.extensions.serving.uri import publish_bundle

    url = publish_bundle(
        args.bundle_dir,
        args.repo_uri,
        private=args.private,
        token=args.token,
        commit_message=args.message,
    )

    json_mode = getattr(args, "json_output", False)
    if json_mode:
        emit_json({"bundle_dir": args.bundle_dir, "repo_uri": args.repo_uri, "commit_url": url})
    else:
        emit_text(f"Published {args.bundle_dir} → {args.repo_uri}\n  commit: {url}")
    return 0

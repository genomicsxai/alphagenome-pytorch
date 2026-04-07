"""agt info — inspect model architecture, heads, tracks, and weight files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from alphagenome_pytorch.cli._output import emit_json, emit_text
from alphagenome_pytorch.extensions.inference.full_chromosome import HEAD_CONFIGS
from alphagenome_pytorch.heads import (
    CONTACT_MAPS_OUTPUT_TRACKS,
    NUM_SPLICE_TISSUES,
    SPLICE_USAGE_OUTPUT_TRACKS,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "info",
        help="Inspect model architecture, heads, tracks, and weight files",
        description="Inspect the model architecture, available heads, track metadata, "
        "and the contents of a weights file.",
    )

    # Optional positional: path to a weights file
    p.add_argument(
        "weights_file",
        nargs="?",
        default=None,
        help="Path to a weights file (.pth or .safetensors) to inspect",
    )

    # Static info flags
    p.add_argument(
        "--heads",
        action="store_true",
        help="List all heads with track counts",
    )

    # Track listing
    p.add_argument(
        "--tracks",
        type=str,
        default=None,
        metavar="HEAD",
        help="List individual tracks for a head",
    )
    p.add_argument(
        "--organism",
        type=str,
        default=None,
        choices=["human", "mouse"],
        help="Organism (default: human)",
    )
    p.add_argument(
        "--search",
        type=str,
        default=None,
        help="Search tracks by substring (matches all metadata fields)",
    )
    p.add_argument(
        "--filter",
        type=str,
        default=None,
        dest="filter_expr",
        help='Filter tracks by metadata field (e.g. "biosample_type=tissue")',
    )
    p.add_argument(
        "--columns",
        type=str,
        default=None,
        help='Columns to display (comma-separated, e.g. "biosample_name,ontology_curie")',
    )
    p.add_argument(
        "--list-fields",
        action="store_true",
        help="List available metadata fields for the selected tracks",
    )

    # Weight file inspection
    p.add_argument(
        "--track-means",
        type=str,
        default=None,
        metavar="HEAD",
        help="Display track_means tensor for a specific head",
    )
    p.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show only top N track means (by value)",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Validate checkpoint — check all keys present, shapes match",
    )
    p.add_argument(
        "--diff",
        type=str,
        default=None,
        metavar="OTHER",
        help="Compare two checkpoints (added/removed/changed keys)",
    )


# ---- All heads including special heads ----

# Resolutions for display
_HEAD_INFO: dict[str, dict[str, Any]] = {}

# Standard genome track heads from HEAD_CONFIGS
for _name, _cfg in HEAD_CONFIGS.items():
    _HEAD_INFO[_name] = {
        "dimension": _cfg["num_tracks"],
        "resolutions": [f"{r}bp" for r in _cfg["resolutions"]],
    }

# Special heads
_HEAD_INFO["contact_maps"] = {
    "dimension": CONTACT_MAPS_OUTPUT_TRACKS,
    "resolutions": ["64x64"],
}
_HEAD_INFO["splice_sites"] = {
    "dimension": 5,
    "resolutions": ["1bp"],
}
_HEAD_INFO["splice_junctions"] = {
    "dimension": NUM_SPLICE_TISSUES * 2,
    "resolutions": ["pairwise"],
}
_HEAD_INFO["splice_site_usage"] = {
    "dimension": SPLICE_USAGE_OUTPUT_TRACKS,
    "resolutions": ["1bp"],
}


def _get_track_counts() -> dict[str, dict[str, int | None]]:
    """Load real (non-padding) track counts from TrackMetadataCatalog."""
    try:
        from alphagenome_pytorch.named_outputs import TrackMetadataCatalog

        counts: dict[str, dict[str, int | None]] = {}
        for org_name, org_val in [("human", 0), ("mouse", 1)]:
            try:
                catalog = TrackMetadataCatalog.load_builtin(org_val)
            except Exception:
                continue
            for head_name in _HEAD_INFO:
                if head_name not in counts:
                    counts[head_name] = {}
                try:
                    tracks = catalog.get_tracks(head_name, organism=org_val)
                    if tracks is not None:
                        real = sum(1 for t in tracks if not t.is_padding)
                        counts[head_name][org_name] = real
                    else:
                        counts[head_name][org_name] = None
                except Exception:
                    counts[head_name][org_name] = None
        return counts
    except Exception:
        return {}


def _run_heads(args: argparse.Namespace) -> int:
    """Display head overview."""
    json_mode = getattr(args, "json_output", False)
    track_counts = _get_track_counts()

    if json_mode:
        heads_list = []
        for name, info in _HEAD_INFO.items():
            entry: dict[str, Any] = {
                "name": name,
                "dimension": info["dimension"],
                "resolutions": info["resolutions"],
            }
            if name in track_counts:
                entry["tracks"] = track_counts[name]
                entry["padding"] = {}
                for org, count in track_counts[name].items():
                    if count is not None:
                        entry["padding"][org] = info["dimension"] - count
            heads_list.append(entry)
        emit_json({"heads": heads_list})
        return 0

    # Text table
    lines: list[str] = []
    header = f"{'Head':<22} {'Tracks (human)':>14} {'Tracks (mouse)':>14} {'Dimension':>9} {'Resolutions'}"
    lines.append(header)

    for name, info in _HEAD_INFO.items():
        tc = track_counts.get(name, {})
        human = str(tc.get("human", "?"))
        mouse = str(tc.get("mouse", "?"))
        dim = str(info["dimension"])
        res = ", ".join(info["resolutions"])
        lines.append(f"{name:<22} {human:>14} {mouse:>14} {dim:>9} {res}")

    emit_text("\n".join(lines))
    return 0


def _track_matches_search(track, query: str) -> bool:
    """Check if any field on a track matches the search query (case-insensitive)."""
    q = query.lower()
    if q in track.track_name.lower():
        return True
    for v in track.extras.values():
        if q in str(v).lower():
            return True
    return False


def _run_tracks(args: argparse.Namespace) -> int:
    """List individual tracks for a head."""
    head_name = args.tracks
    org_name = args.organism or "human"
    org_idx = 0 if org_name == "human" else 1
    json_mode = getattr(args, "json_output", False)

    if head_name not in _HEAD_INFO:
        print(f"Error: Unknown head '{head_name}'. Available: {list(_HEAD_INFO.keys())}", file=sys.stderr)
        return 1

    dim = _HEAD_INFO[head_name]["dimension"]

    try:
        from alphagenome_pytorch.named_outputs import TrackMetadataCatalog
        catalog = TrackMetadataCatalog.load_builtin(org_idx)
        tracks = catalog.get_tracks(head_name, organism=org_idx)
    except Exception as exc:
        print(f"Error: Could not load track metadata: {exc}", file=sys.stderr)
        print("Install pandas for track metadata: pip install pandas", file=sys.stderr)
        return 1

    if tracks is None:
        print(f"Error: No tracks found for head '{head_name}'", file=sys.stderr)
        return 1

    # Separate real and padding tracks
    real_tracks = [t for t in tracks if not t.is_padding]
    num_real = len(real_tracks)
    num_padding = dim - num_real

    # Apply search (matches any field) and filter (exact field=value)
    filtered = real_tracks
    if args.search:
        filtered = [t for t in filtered if _track_matches_search(t, args.search)]

    if args.filter_expr:
        if "=" not in args.filter_expr:
            print("Error: --filter must be in 'field=value' format", file=sys.stderr)
            return 1
        field, value = args.filter_expr.split("=", 1)
        field, value = field.strip(), value.strip()
        filtered = [t for t in filtered if str(t.get(field, "")) == value]

    if json_mode:
        track_list = []
        for t in filtered:
            entry = {"index": t.track_index, "name": t.track_name}
            if t.extras:
                entry.update(t.extras)
            track_list.append(entry)
        emit_json({
            "head": head_name,
            "organism": org_name,
            "tracks": track_list,
            "total_tracks": num_real,
            "showing": len(filtered),
            "dimension": dim,
            "padding": num_padding,
        })
        return 0

    # Text output
    is_filtered = args.search or args.filter_expr

    header_parts = [f"Head: {head_name} | {num_real} tracks / {dim} dimension ({num_padding} padding) | {org_name}"]
    if args.search:
        header_parts.append(f"Search: '{args.search}' → {len(filtered)} match(es)")
    if args.filter_expr:
        header_parts.append(f"Filter: {args.filter_expr} → {len(filtered)} match(es)")

    lines = header_parts + [""]

    # Tabular format — compact default columns; --columns to override/expand
    display_cols = _pick_display_columns(
        filtered, head_name,
        user_columns=getattr(args, 'columns', None),
    )

    if display_cols:
        col_header = f"  {'#':>3}   {'biosample_name':<28}"
        for col in display_cols:
            col_header += f" {col:<20}"
        lines.append(col_header)
        lines.append("  " + "─" * (len(col_header) - 2))

    for t in filtered:
        biosample = t.get("biosample_name", "")
        line = f"  {t.track_index:>3}   {biosample:<28}"
        for col in display_cols:
            val = str(t.get(col, "")) if t.get(col) is not None else ""
            if len(val) > 20:
                val = val[:17] + "..."
            line += f" {val:<20}"
        lines.append(line)

    if num_padding > 0 and not is_filtered:
        lines.append(f"\n  ({num_padding} padding tracks omitted)")

    emit_text("\n".join(lines))
    return 0


_DEFAULT_DISPLAY_COLUMNS: tuple[str, ...] = ("biosample_type", "ontology_curie")

# Head-specific column shown in addition to the defaults
_HEAD_EXTRA_COLUMN: dict[str, str] = {
    "chip_tf": "transcription_factor",
    "chip_histone": "histone_mark",
}


def _pick_display_columns(
    tracks: list,
    head_name: str,
    user_columns: str | None = None,
) -> list[str]:
    """Pick extra columns to display for this head.

    Default is a small fixed set (biosample_type, ontology_curie, plus a
    head-specific column for chip_tf/chip_histone). Pass --columns to
    override.
    """
    if not tracks:
        return []

    # User-specified columns take priority. biosample_name is always shown
    # as the leading column, so drop it from extras to avoid duplicates.
    if user_columns:
        return [
            c.strip() for c in user_columns.split(",")
            if c.strip() and c.strip() != "biosample_name"
        ]

    cols: list[str] = []
    if head_name in _HEAD_EXTRA_COLUMN:
        cols.append(_HEAD_EXTRA_COLUMN[head_name])
    cols.extend(_DEFAULT_DISPLAY_COLUMNS)

    # Drop columns that aren't present on any track
    available: set[str] = set()
    for t in tracks:
        available.update(t.extras.keys())
    return [c for c in cols if c in available]


def _load_state_dict(path: str) -> tuple[dict, str]:
    """Load a state dict from .pth or .safetensors file.

    Returns (state_dict, format_name).
    """
    p = Path(path)
    if p.suffix == ".safetensors" or p.suffixes[-2:] == [".delta", ".safetensors"]:
        from safetensors.torch import load_file
        return load_file(str(p)), "safetensors"
    else:
        import torch
        return torch.load(str(p), map_location="cpu", weights_only=True), "pth"


def _run_weights(args: argparse.Namespace) -> int:
    """Inspect a weights file."""
    path = args.weights_file
    json_mode = getattr(args, "json_output", False)

    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}")

    state_dict, fmt = _load_state_dict(path)
    file_size_mb = os.path.getsize(path) / 1e6

    # Count parameters
    total_params = 0
    dtypes = set()
    head_names = set()
    has_track_means = False

    for key, tensor in state_dict.items():
        total_params += tensor.numel()
        dtypes.add(str(tensor.dtype))
        if key.startswith("heads."):
            parts = key.split(".")
            if len(parts) >= 2:
                head_names.add(parts[1])
        if key.endswith(".track_means"):
            has_track_means = True

    dtype_str = ", ".join(sorted(dtypes))
    sorted_heads = sorted(head_names)

    # Handle --track-means
    if args.track_means:
        return _run_track_means(args, state_dict)

    # Handle --validate
    if args.validate:
        return _run_validate(args, state_dict)

    # Handle --diff
    if args.diff:
        return _run_diff(args, state_dict, fmt)

    if json_mode:
        emit_json({
            "file": path,
            "format": fmt,
            "file_size_mb": round(file_size_mb, 1),
            "total_parameters": total_params,
            "dtype": dtype_str,
            "has_track_means": has_track_means,
            "heads": sorted_heads,
        })
    else:
        lines = [
            f"File: {path}",
            f"Format: {fmt}",
            f"Size: {file_size_mb:.1f} MB",
            f"Parameters: {total_params:,}",
            f"Dtype: {dtype_str}",
            f"Track means: {'yes' if has_track_means else 'no'}",
            f"Heads: {', '.join(sorted_heads) if sorted_heads else '(none)'}",
        ]
        emit_text("\n".join(lines))

    return 0


def _run_track_means(args: argparse.Namespace, state_dict: dict) -> int:
    """Show track_means for a given head."""
    head = args.track_means
    org_name = args.organism or "human"
    org_idx = 0 if org_name == "human" else 1
    json_mode = getattr(args, "json_output", False)

    key = f"heads.{head}.track_means"
    if key not in state_dict:
        print(f"Error: No track_means found for head '{head}'", file=sys.stderr)
        return 1

    means = state_dict[key]
    if means.dim() >= 2:
        means = means[org_idx]

    top_n = args.top
    if top_n is not None:
        # Sort descending, take top N
        values, indices = means.sort(descending=True)
        values = values[:top_n]
        indices = indices[:top_n]
    else:
        indices = list(range(len(means)))
        values = means

    if json_mode:
        entries = [
            {"index": int(idx), "mean": float(val)}
            for idx, val in zip(indices, values)
        ]
        emit_json({
            "head": head,
            "organism": org_name,
            "track_means": entries,
        })
    else:
        lines = [f"Track means for {head} ({org_name}):"]
        for idx, val in zip(indices, values):
            lines.append(f"  {int(idx):>5}  {float(val):.6f}")
        emit_text("\n".join(lines))

    return 0


def _run_validate(args: argparse.Namespace, state_dict: dict) -> int:
    """Validate checkpoint against model architecture."""
    json_mode = getattr(args, "json_output", False)
    import torch
    from alphagenome_pytorch import AlphaGenome

    model = AlphaGenome(num_organisms=2)
    expected = model.state_dict()

    missing = [k for k in expected if k not in state_dict]
    unexpected = [k for k in state_dict if k not in expected]
    mismatched = []
    for k in expected:
        if k in state_dict and expected[k].shape != state_dict[k].shape:
            mismatched.append({
                "key": k,
                "expected": list(expected[k].shape),
                "actual": list(state_dict[k].shape),
            })

    ok = not missing and not unexpected and not mismatched

    if json_mode:
        emit_json({
            "valid": ok,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "shape_mismatches": mismatched,
        })
    else:
        if ok:
            emit_text("✓ Checkpoint is valid — all keys present and shapes match.")
        else:
            lines = []
            if missing:
                lines.append(f"Missing keys ({len(missing)}):")
                for k in missing[:20]:
                    lines.append(f"  - {k}")
                if len(missing) > 20:
                    lines.append(f"  ... and {len(missing) - 20} more")
            if unexpected:
                lines.append(f"Unexpected keys ({len(unexpected)}):")
                for k in unexpected[:20]:
                    lines.append(f"  - {k}")
                if len(unexpected) > 20:
                    lines.append(f"  ... and {len(unexpected) - 20} more")
            if mismatched:
                lines.append(f"Shape mismatches ({len(mismatched)}):")
                for m in mismatched[:20]:
                    lines.append(f"  - {m['key']}: expected {m['expected']}, got {m['actual']}")
            emit_text("\n".join(lines))

    return 0 if ok else 1


def _run_diff(args: argparse.Namespace, state_dict: dict, fmt: str) -> int:
    """Compare two checkpoints."""
    json_mode = getattr(args, "json_output", False)
    other_dict, other_fmt = _load_state_dict(args.diff)

    keys_a = set(state_dict.keys())
    keys_b = set(other_dict.keys())

    added = sorted(keys_b - keys_a)
    removed = sorted(keys_a - keys_b)
    changed = []
    for k in sorted(keys_a & keys_b):
        if state_dict[k].shape != other_dict[k].shape:
            changed.append({"key": k, "reason": "shape", "a": list(state_dict[k].shape), "b": list(other_dict[k].shape)})
        elif not (state_dict[k] == other_dict[k]).all():
            changed.append({"key": k, "reason": "values"})

    if json_mode:
        emit_json({
            "added": added,
            "removed": removed,
            "changed": changed,
        })
    else:
        lines = []
        if added:
            lines.append(f"Added ({len(added)}):")
            for k in added[:30]:
                lines.append(f"  + {k}")
        if removed:
            lines.append(f"Removed ({len(removed)}):")
            for k in removed[:30]:
                lines.append(f"  - {k}")
        if changed:
            lines.append(f"Changed ({len(changed)}):")
            for c in changed[:30]:
                if c["reason"] == "shape":
                    lines.append(f"  ~ {c['key']} ({c['a']} → {c['b']})")
                else:
                    lines.append(f"  ~ {c['key']} (values differ)")
        if not added and not removed and not changed:
            lines.append("Checkpoints are identical.")
        emit_text("\n".join(lines))

    return 0


def _run_list_fields(args: argparse.Namespace) -> int:
    """List available metadata fields for one or all heads."""
    from alphagenome_pytorch.named_outputs import TrackMetadataCatalog

    org_name = args.organism or "human"
    org_idx = 0 if org_name == "human" else 1
    head_name = args.tracks  # may be None

    try:
        catalog = TrackMetadataCatalog.load_builtin(org_idx)
    except Exception as exc:
        print(f"Error: Could not load track metadata: {exc}", file=sys.stderr)
        return 1

    if head_name:
        if head_name not in _HEAD_INFO:
            print(f"Error: Unknown head '{head_name}'. Available: {list(_HEAD_INFO.keys())}", file=sys.stderr)
            return 1
        heads = [head_name]
    else:
        heads = list(_HEAD_INFO.keys())

    all_fields: set[str] = set()
    for h in heads:
        tracks = catalog.get_tracks(h, organism=org_idx)
        if tracks:
            for t in tracks:
                all_fields.update(t.extras.keys())

    label = f"for {head_name}" if head_name else "across all heads"
    lines = [f"Available fields {label} ({org_name}):"]
    for f in sorted(all_fields):
        lines.append(f"  {f}")

    usage_head = head_name or "atac"
    lines.append(f"\nUsage: agt info --tracks {usage_head} --columns biosample_type,ontology_curie")
    emit_text("\n".join(lines))
    return 0


def run(args: argparse.Namespace) -> int:
    """Dispatch info subcommand."""
    # List fields globally if requested
    if getattr(args, "list_fields", False):
        return _run_list_fields(args)

    if args.tracks:
        return _run_tracks(args)
    if args.weights_file:
        return _run_weights(args)
    # Default: show heads overview
    return _run_heads(args)

#!/usr/bin/env python
"""Evaluate AlphaGenome checkpoints.

Supports three opt-in features (--metrics, --regions, --ism) and an optional
native-head comparison layer (--native-biosample) that enriches all outputs.

Usage examples:

    # Metrics only
    python scripts/evaluate_checkpoint.py \
        --checkpoint best_model.pth \
        --pretrained-weights model_fold1.pth \
        --genome GRCh38.fa --bigwig signal.bw \
        --test-bed test.bed --output-dir eval/ --metrics

    # Metrics + native comparison
    python scripts/evaluate_checkpoint.py \
        --checkpoint best_model.pth \
        --pretrained-weights model_fold1.pth \
        --genome GRCh38.fa --bigwig signal.bw \
        --test-bed test.bed --output-dir eval/ --metrics \
        --native-biosample "WTC11"

    # Native/pretrained model metrics only
    python scripts/evaluate_checkpoint.py \
        --native-only --metrics \
        --pretrained-weights model_fold1.pth \
        --genome GRCh38.fa --bigwig signal.bw \
        --test-bed test.bed --output-dir eval_native/ \
        --modality atac --native-biosample "K562" \
        --sequence-length 131072 --resolutions 128

    # Regions + ISM
    python scripts/evaluate_checkpoint.py \
        --checkpoint best_model.pth \
        --pretrained-weights model_fold1.pth \
        --genome GRCh38.fa --bigwig signal.bw \
        --regions-bed regions.bed --output-dir eval/ \
        --regions --ism --ism-window-size 21

    # Everything
    python scripts/evaluate_checkpoint.py \
        --checkpoint best_model.pth \
        --pretrained-weights model_fold1.pth \
        --genome GRCh38.fa --bigwig signal.bw \
        --test-bed test.bed --regions-bed regions.bed \
        --output-dir eval/ --metrics --regions --ism \
        --native-biosample "WTC11"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from alphagenome_pytorch.extensions.finetuning.checkpointing import (
    load_finetuned_model as _load_finetuned_model,
)
from alphagenome_pytorch.extensions.finetuning.datasets import GenomicDataset
from alphagenome_pytorch.extensions.finetuning.evaluation import (
    collect_targets,
    compute_all_metrics,
    evaluate_native_split,
    evaluate_split,
    load_native_model,
)
from alphagenome_pytorch.extensions.finetuning.training import collate_genomic

log = logging.getLogger(__name__)


def load_finetuned_model(
    checkpoint_path: str, pretrained_weights: str, device: torch.device,
) -> tuple[nn.Module, dict]:
    """Load a finetuned checkpoint, auto-detecting delta/full/adapter format.

    Full checkpoints embed their TransferConfig (lora/locon targets etc.) at
    save time, so adapter checkpoints are self-describing -- no external
    transfer config file is needed.
    """
    model, meta = _load_finetuned_model(
        checkpoint_path=checkpoint_path,
        pretrained_weights=pretrained_weights,
        device=device,
        merge=True,
    )
    for p in model.parameters():
        p.requires_grad = False
    return model, meta


# =============================================================================
# Plotting
# =============================================================================


def plot_scatter(
    preds: np.ndarray, targets: np.ndarray,
    out_path: Path, title_suffix: str = "",
) -> None:
    """Scatter plot of predicted vs observed (subsampled)."""
    p = preds.flatten()
    t = targets.flatten()
    n = min(len(p), 100_000)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(p), n, replace=False)
    p, t = p[idx], t[idx]
    r = stats.pearsonr(p, t)[0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t, p, alpha=0.05, s=1, color="steelblue", rasterized=True)
    lim = max(t.max(), p.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Observed signal")
    ax.set_ylabel("Predicted signal")
    ax.set_title(f"Pred vs Obs {title_suffix} (r={r:.3f})")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter_counts(
    preds: np.ndarray, targets: np.ndarray,
    out_path: Path, title_suffix: str = "",
) -> None:
    """Scatter plot of total counts per region."""
    pred_sums = preds.sum(axis=1).flatten()
    target_sums = targets.sum(axis=1).flatten()
    r = stats.pearsonr(pred_sums, target_sums)[0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        target_sums, pred_sums, alpha=0.15, s=5, color="steelblue", rasterized=True,
    )
    lim = max(target_sums.max(), pred_sums.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Observed total count")
    ax.set_ylabel("Predicted total count")
    ax.set_title(f"Count correlation {title_suffix} (r={r:.3f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlation_histogram(
    ft_values: np.ndarray,
    out_path: Path,
    native_values: np.ndarray | None = None,
    xlabel: str = "Pearson r (per region)",
    title: str = "Per-region correlation distribution",
    ft_label: str = "Finetuned",
    native_label: str = "Native",
) -> None:
    """Histogram of per-region values, optionally overlaid with native."""
    fig, ax = plt.subplots(figsize=(7, 4))

    bins = np.linspace(
        min(ft_values.min(), native_values.min() if native_values is not None else ft_values.min()),
        max(ft_values.max(), native_values.max() if native_values is not None else ft_values.max()),
        51,
    )

    ax.hist(
        ft_values, bins=bins, alpha=0.6, color="steelblue",
        edgecolor="white", label=f"{ft_label} (med={np.median(ft_values):.3f})",
    )
    if native_values is not None:
        ax.hist(
            native_values, bins=bins, alpha=0.5, color="forestgreen",
            edgecolor="white", label=f"{native_label} (med={np.median(native_values):.3f})",
        )

    ax.axvline(
        np.median(ft_values), color="steelblue", linestyle="--", linewidth=1.2,
    )
    if native_values is not None:
        ax.axvline(
            np.median(native_values), color="forestgreen", linestyle="--", linewidth=1.2,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_region_tracks(
    ft_pred: np.ndarray,
    target: np.ndarray,
    region_name: str,
    out_path: Path,
    res: int,
    native_pred: np.ndarray | None = None,
    ft_r: float | None = None,
    native_r: float | None = None,
) -> None:
    """Plot observed vs predicted signal for one region.

    All arrays are 1D (seq_len,) — single track.
    """
    fig, ax = plt.subplots(figsize=(14, 3))
    x = np.arange(len(target))

    ax.fill_between(x, target, alpha=0.3, color="steelblue", label="Observed")
    label_ft = "Finetuned"
    if ft_r is not None:
        label_ft += f" (r={ft_r:.3f})"
    ax.plot(x, ft_pred, color="crimson", linewidth=0.8, alpha=0.8, label=label_ft)

    if native_pred is not None:
        label_nat = "Native"
        if native_r is not None:
            label_nat += f" (r={native_r:.3f})"
        ax.plot(
            x, native_pred, color="forestgreen", linewidth=0.8,
            alpha=0.7, linestyle="--", label=label_nat,
        )

    ax.set_title(f"{region_name} ({res}bp)", fontsize=10)
    ax.set_xlabel(f"Position ({res}bp bins)" if res > 1 else "Position (bp)")
    ax.set_ylabel("Signal")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ism_heatmap(
    ism_matrix: np.ndarray,
    region_name: str,
    out_path: Path,
    center_pos: int,
) -> None:
    """Plot ISM heatmap (position x 4 nucleotides).

    Args:
        ism_matrix: (window_size, 4) array of ISM scores.
        region_name: Name for the title.
        out_path: Output path.
        center_pos: Genomic center position (1-based) for axis label.
    """
    fig, ax = plt.subplots(figsize=(max(6, ism_matrix.shape[0] * 0.4), 3))
    vmax = np.abs(ism_matrix).max()
    im = ax.imshow(
        ism_matrix.T, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    ax.set_yticks(range(4))
    ax.set_yticklabels(["A", "C", "G", "T"])
    half = ism_matrix.shape[0] // 2
    ax.set_xlabel(f"Position (centered on {center_pos})")
    ax.set_title(f"ISM: {region_name}")
    plt.colorbar(im, ax=ax, label="Effect size")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Region parsing
# =============================================================================


def parse_regions_bed(bed_path: str) -> list[dict]:
    """Parse BED4 file into list of region dicts.

    Returns list of {chrom, start, end, name, midpoint}.
    """
    regions = []
    with open(bed_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.split("\t")
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            name = parts[3] if len(parts) > 3 else f"{chrom}:{start}-{end}"
            midpoint = (start + end) // 2
            regions.append({
                "chrom": chrom,
                "start": start,
                "end": end,
                "name": name,
                "midpoint": midpoint,
            })
    log.info("Loaded %d regions from %s", len(regions), bed_path)
    return regions


# =============================================================================
# ISM
# =============================================================================


def run_ism_for_regions(
    model: nn.Module,
    genome_path: str,
    regions: list[dict],
    modality: str,
    ism_window_size: int,
    device: torch.device,
    out_dir: Path,
) -> None:
    """Run ISM on each region and save heatmap plots."""
    from alphagenome_pytorch.variant_scoring.aggregations import AggregationType
    from alphagenome_pytorch.variant_scoring.inference import VariantScoringModel
    from alphagenome_pytorch.variant_scoring.scorers.center_mask import (
        CenterMaskScorer,
    )
    from alphagenome_pytorch.variant_scoring.types import Interval, OutputType

    out_dir.mkdir(parents=True, exist_ok=True)

    scoring_model = VariantScoringModel(
        model=model, fasta_path=genome_path, device=device,
    )

    output_type = OutputType(modality)
    scorer = CenterMaskScorer(
        output_type, width=501, aggregation_type=AggregationType.DIFF_LOG2_SUM,
    )

    for region in tqdm(regions, desc="ISM"):
        name = region["name"]
        center = region["midpoint"]  # 0-based midpoint
        center_1based = center + 1

        # Create 131kb interval centered on the region
        interval = Interval.centered_on(
            region["chrom"], center_1based, width=131072,
        )

        try:
            ism_results = scoring_model.score_ism_variants(
                interval=interval,
                center_position=center_1based,
                scorers=[scorer],
                window_size=ism_window_size,
                nucleotides="ACGT",
                progress=False,
            )
        except Exception as e:
            log.warning("ISM failed for %s: %s", name, e)
            continue

        # Extract scores and variants for the single scorer
        scores = []
        variants = []
        for variant_results in ism_results:
            # variant_results is per-scorer: single scorer → index 0
            vs = variant_results[0]
            scores.append(vs.scores.mean().item())  # Average across tracks
            variants.append(vs.variant)

        if not scores:
            log.warning("No ISM scores for %s", name)
            continue

        # Build ISM matrix
        matrix = scoring_model.ism_matrix(
            variant_scores=scores,
            variants=variants,
            interval=Interval.centered_on(
                region["chrom"], center_1based, width=ism_window_size,
            ),
            multiply_by_sequence=True,
        )
        matrix_np = matrix.numpy()

        plot_ism_heatmap(
            matrix_np, name, out_dir / f"{name}_ism.png", center_1based,
        )

    log.info("ISM heatmaps saved to %s", out_dir)


# =============================================================================
# Summary
# =============================================================================


def format_summary_table(
    ft_metrics: dict | None,
    native_metrics: dict | None,
    native_display_name: str | None,
    resolutions: tuple[int, ...],
) -> str:
    """Format a human-readable summary table."""
    lines = []
    for res in resolutions:
        ft = ft_metrics.get(res) if ft_metrics else None
        nat = native_metrics.get(res) if native_metrics else None
        if ft is None and nat is None:
            continue

        lines.append(f"\n--- {res}bp resolution ---")
        header = f"{'Metric':<28}"
        if ft is not None:
            header += f"{'Finetuned':>12}"
        if nat is not None:
            header += f"  {'Native(' + (native_display_name or '?') + ')':>20}"
        lines.append(header)
        lines.append("-" * len(header))

        rows = [
            ("Profile r (mean)", "profile_pearson_r_mean"),
            ("Profile r (median)", "profile_pearson_r_median"),
            ("Count r", "count_pearson_r"),
            ("JSD (mean)", "jsd_mean"),
            ("JSD (median)", "jsd_median"),
            ("MSE", "mse"),
            ("Spearman (global)", "spearman_global"),
        ]
        for label, key in rows:
            line = f"{label:<28}"
            if ft is not None:
                line += f"{ft[key]:>12.4f}"
            if nat is not None:
                line += f"  {nat[key]:>20.4f}"
            lines.append(line)
        line = f"{'N regions':<28}"
        if ft is not None:
            line += f"{ft['n_regions']:>12d}"
        if nat is not None:
            line += f"  {nat['n_regions']:>20d}"
        lines.append(line)

    return "\n".join(lines)


def save_summary_json(
    ft_metrics: dict | None,
    native_metrics: dict | None,
    checkpoint_meta: dict,
    native_info: dict | None,
    loss: float | None,
    out_path: Path,
) -> None:
    """Save machine-readable JSON summary."""
    def _clean(m: dict) -> dict:
        """Remove numpy arrays, keep scalars."""
        return {
            k: v for k, v in m.items()
            if not isinstance(v, np.ndarray)
        }

    data: dict = {"checkpoint": checkpoint_meta}
    if loss is not None:
        data["loss"] = loss

    if ft_metrics:
        data["finetuned"] = {
            str(res): _clean(m) for res, m in ft_metrics.items()
        }
    if native_metrics:
        data["native"] = {
            str(res): _clean(m) for res, m in native_metrics.items()
        }
    if native_info:
        data["native_track"] = native_info

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate AlphaGenome checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    p.add_argument("--checkpoint", help="Finetuned checkpoint path")
    p.add_argument(
        "--pretrained-weights", required=True,
        help="Pretrained trunk weights (e.g. model_fold1.pth)",
    )
    p.add_argument("--output-dir", required=True, help="Output directory")

    # Feature flags
    p.add_argument(
        "--metrics", action="store_true",
        help="Compute test set metrics (needs --test-bed + --bigwig)",
    )
    p.add_argument(
        "--regions", action="store_true",
        help="Plot predefined regions (needs --regions-bed + --bigwig)",
    )
    p.add_argument(
        "--ism", action="store_true",
        help="Run ISM on predefined regions (needs --regions-bed + --genome)",
    )
    p.add_argument(
        "--native-only", action="store_true",
        help=(
            "Evaluate the pretrained/native model directly. Requires --metrics, "
            "--modality, and --native-biosample or --native-track-index."
        ),
    )

    # Data inputs
    p.add_argument("--genome", help="Reference genome FASTA")
    p.add_argument("--bigwig", nargs="+", help="BigWig signal file(s)")
    p.add_argument("--test-bed", help="Test split BED file")
    p.add_argument("--regions-bed", help="Named regions BED4 file (chrom, start, end, name)")

    # Native head comparison
    p.add_argument(
        "--native-biosample",
        help="Biosample name for native head comparison (e.g. 'WTC11')",
    )
    p.add_argument(
        "--native-track-index", type=int,
        help="Direct track index for native head (alternative to --native-biosample)",
    )
    p.add_argument(
        "--modality",
        help="Native head/modality to evaluate in --native-only mode (e.g. atac)",
    )

    # Options
    p.add_argument(
        "--resolutions",
        default=None,
        help="Comma-separated resolutions for --native-only mode (default: 128)",
    )
    p.add_argument("--sequence-length", type=int, default=131072)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-regions", type=int, default=2000,
                    help="Max regions for metrics (0=all)")
    p.add_argument("--ism-window-size", type=int, default=21)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--device", default=None, help="Device (default: auto)")

    return p.parse_args()


def parse_resolutions(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return (128,)
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s",
    )
    args = parse_args()

    # Resolve device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output: %s", out_dir)
    log.info("Device: %s", device)

    # Determine which features to run
    any_explicit = args.metrics or args.regions or args.ism or args.native_only
    run_metrics = args.metrics or (not any_explicit and args.test_bed is not None)
    run_regions = args.regions or (not any_explicit and args.regions_bed is not None)
    run_ism = args.ism  # Always explicit
    want_native = bool(args.native_biosample or args.native_track_index is not None)

    # Validate inputs
    if not args.native_only and not args.checkpoint:
        sys.exit("Error: --checkpoint is required unless --native-only is set")
    if args.native_only and not args.metrics:
        sys.exit("Error: --native-only currently supports metrics only; pass --metrics")
    if args.native_only and (run_regions or run_ism):
        sys.exit("Error: --native-only does not support --regions or --ism")
    if args.native_only and not args.modality:
        sys.exit("Error: --native-only requires --modality")
    if args.native_only and not want_native:
        sys.exit("Error: --native-only requires --native-biosample or --native-track-index")
    if run_metrics and (not args.test_bed or not args.bigwig):
        sys.exit("Error: --metrics requires --test-bed and --bigwig")
    if args.native_only and args.bigwig and len(args.bigwig) != 1:
        sys.exit("Error: --native-only evaluates one native track; pass exactly one --bigwig")
    if run_regions and (not args.regions_bed or not args.bigwig):
        sys.exit("Error: --regions requires --regions-bed and --bigwig")
    if run_ism and (not args.regions_bed or not args.genome):
        sys.exit("Error: --ism requires --regions-bed and --genome")

    model = None
    if args.native_only:
        modality = args.modality
        resolutions = parse_resolutions(args.resolutions)
        track_names = []
        ckpt_meta = {
            "mode": "native_only",
            "pretrained_weights": args.pretrained_weights,
            "modality": modality,
            "resolutions": resolutions,
            "track_names": track_names,
        }
        log.info(
            "Native-only modality: %s, Resolutions: %s",
            modality, resolutions,
        )
    else:
        # ---- Load finetuned model ----
        model, ckpt_meta = load_finetuned_model(
            args.checkpoint, args.pretrained_weights, device,
        )
        modality = ckpt_meta["modality"]
        if isinstance(modality, list):
            modality = modality[0]  # single-task evaluation
        resolutions = ckpt_meta["resolutions"]
        if isinstance(resolutions, dict):
            resolutions = resolutions.get(modality, (1,))
        resolutions = tuple(resolutions)
        track_names = ckpt_meta["track_names"]
        if isinstance(track_names, dict):
            track_names = track_names.get(modality, [])

        log.info(
            "Modality: %s, Tracks: %s, Resolutions: %s",
            modality, track_names, resolutions,
        )
        log.info(
            "Checkpoint epoch: %s, val_loss: %s",
            ckpt_meta["epoch"], ckpt_meta["val_loss"],
        )

    # ---- Optionally load native model ----
    native_model, native_track_idx, native_display_name = None, None, None
    if want_native:
        native_model, native_track_idx, native_display_name = load_native_model(
            args.pretrained_weights, args.native_biosample,
            args.native_track_index, modality, device,
        )

    # ---- Feature: metrics ----
    ft_metrics_by_res: dict[int, dict] = {}
    native_metrics_by_res: dict[int, dict] = {}
    loss = None

    if run_metrics:
        log.info("=" * 60)
        log.info("Computing test set metrics")
        log.info("=" * 60)

        metrics_dir = out_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)

        dataset = GenomicDataset(
            genome_fasta=args.genome,
            bigwig_files=args.bigwig,
            bed_file=args.test_bed,
            resolutions=resolutions,
            sequence_length=args.sequence_length,
        )

        max_regions = args.max_regions if args.max_regions > 0 else None
        if max_regions and len(dataset) > max_regions:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(dataset), max_regions, replace=False).tolist()
            dataset_sub = Subset(dataset, indices)
            positions = [dataset._positions_list[i] for i in indices]
        else:
            dataset_sub = dataset
            positions = dataset._positions_list

        loader = DataLoader(
            dataset_sub,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_genomic,
        )

        ft_preds: dict[int, np.ndarray] = {}
        native_preds: dict[int, np.ndarray] = {}
        targets: dict[int, np.ndarray]

        if args.native_only:
            targets = collect_targets(loader, resolutions)
        else:
            # Finetuned evaluation
            ft_preds, targets, loss = evaluate_split(
                model, modality, loader, device, resolutions,
            )
            log.info("Loss: %.4f", loss)

            for res in resolutions:
                if res not in ft_preds:
                    continue
                p, t = ft_preds[res], targets[res]
                log.info("Resolution %dbp: preds %s, targets %s", res, p.shape, t.shape)

                ft_m = compute_all_metrics(p, t)
                ft_metrics_by_res[res] = ft_m

                log.info("  Profile r (mean):  %.4f", ft_m["profile_pearson_r_mean"])
                log.info("  Profile r (median):%.4f", ft_m["profile_pearson_r_median"])
                log.info("  Count r:           %.4f", ft_m["count_pearson_r"])
                log.info("  JSD (mean):        %.4f", ft_m["jsd_mean"])

                # Scatter plots
                plot_scatter(
                    p, t, metrics_dir / f"scatter_test_{res}bp.png",
                    title_suffix=f"(test, {res}bp)",
                )
                plot_scatter_counts(
                    p, t, metrics_dir / f"scatter_counts_test_{res}bp.png",
                    title_suffix=f"(test, {res}bp)",
                )

        # Native evaluation on same data
        if native_model is not None:
            # Move finetuned model to CPU to free GPU memory.
            model_device = device
            if model is not None:
                model.cpu()
            native_model = native_model.to(device)

            native_preds = evaluate_native_split(
                native_model, modality, native_track_idx,
                loader, device, resolutions,
            )

            for res in resolutions:
                if res not in native_preds or res not in targets:
                    continue
                nat_p = native_preds[res]
                t = targets[res]
                nat_m = compute_all_metrics(nat_p, t)
                native_metrics_by_res[res] = nat_m

                log.info(
                    "Native %dbp — Profile r: %.4f, Count r: %.4f, JSD: %.4f",
                    res, nat_m["profile_pearson_r_mean"],
                    nat_m["count_pearson_r"], nat_m["jsd_mean"],
                )

            # Move models back
            native_model.cpu()
            if model is not None:
                model = model.to(model_device)

        # Generate histogram plots (overlaid if native available)
        for res in resolutions:
            ft_m = ft_metrics_by_res.get(res)
            nat_m = native_metrics_by_res.get(res)
            if ft_m is None and nat_m is None:
                continue
            if ft_m is None:
                plot_correlation_histogram(
                    nat_m["profile_pearson_r_all"],
                    metrics_dir / f"native_correlation_hist_test_{res}bp.png",
                    xlabel="Pearson r (per region)",
                    title=f"Native profile correlation distribution ({res}bp)",
                    ft_label=f"Native ({native_display_name})" if native_display_name else "Native",
                )
                plot_correlation_histogram(
                    nat_m["jsd_all"],
                    metrics_dir / f"native_jsd_hist_test_{res}bp.png",
                    xlabel="JSD (per region)",
                    title=f"Native JSD distribution ({res}bp)",
                    ft_label=f"Native ({native_display_name})" if native_display_name else "Native",
                )
                continue

            plot_correlation_histogram(
                ft_m["profile_pearson_r_all"],
                metrics_dir / f"correlation_hist_test_{res}bp.png",
                native_values=nat_m["profile_pearson_r_all"] if nat_m else None,
                xlabel="Pearson r (per region)",
                title=f"Profile correlation distribution ({res}bp)",
                native_label=f"Native ({native_display_name})" if native_display_name else "Native",
            )
            plot_correlation_histogram(
                ft_m["jsd_all"],
                metrics_dir / f"jsd_hist_test_{res}bp.png",
                native_values=nat_m["jsd_all"] if nat_m else None,
                xlabel="JSD (per region)",
                title=f"JSD distribution ({res}bp)",
                native_label=f"Native ({native_display_name})" if native_display_name else "Native",
            )

        # Save predictions
        if args.save_predictions:
            pred_dir = out_dir / "predictions"
            pred_dir.mkdir(exist_ok=True)
            bed_out = pred_dir / "test_regions.bed"
            with open(bed_out, "w") as f:
                for chrom, start, end in positions:
                    f.write(f"{chrom}\t{start}\t{end}\n")
            for res in resolutions:
                if res in ft_preds:
                    np.save(pred_dir / f"test_preds_{res}bp.npy", ft_preds[res].astype(np.float16))
                    np.save(pred_dir / f"test_targets_{res}bp.npy", targets[res].astype(np.float16))
                if native_model is not None and res in native_preds:
                    np.save(pred_dir / f"native_preds_{res}bp.npy", native_preds[res].astype(np.float16))

    # ---- Feature: region exploration ----
    if run_regions:
        log.info("=" * 60)
        log.info("Region exploration")
        log.info("=" * 60)

        regions_dir = out_dir / "regions"
        regions_dir.mkdir(exist_ok=True)

        regions = parse_regions_bed(args.regions_bed)

        # Create dataset from regions BED
        region_dataset = GenomicDataset(
            genome_fasta=args.genome,
            bigwig_files=args.bigwig,
            bed_file=args.regions_bed,
            resolutions=resolutions,
            sequence_length=args.sequence_length,
        )
        region_loader = DataLoader(
            region_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_genomic,
        )

        # Finetuned predictions on regions
        ft_region_preds, region_targets, _ = evaluate_split(
            model, modality, region_loader, device, resolutions,
        )

        # Native predictions on regions
        native_region_preds = None
        if native_model is not None:
            model.cpu()
            native_model = native_model.to(device)
            native_region_preds = evaluate_native_split(
                native_model, modality, native_track_idx,
                region_loader, device, resolutions,
            )
            native_model.cpu()
            model = model.to(device)

        # Plot each region
        for res in resolutions:
            if res not in ft_region_preds:
                continue
            ft_p = ft_region_preds[res]
            t = region_targets[res]
            nat_p = native_region_preds[res] if native_region_preds and res in native_region_preds else None

            for i, region in enumerate(regions):
                if i >= len(ft_p):
                    break
                # Use first track for plotting
                ft_track = ft_p[i, :, 0]
                obs_track = t[i, :, 0]
                nat_track = nat_p[i, :, 0] if nat_p is not None else None

                # Per-region correlation
                ft_r = stats.pearsonr(ft_track, obs_track)[0] if np.std(obs_track) > 1e-10 else 0.0
                nat_r = None
                if nat_track is not None and np.std(obs_track) > 1e-10:
                    nat_r = stats.pearsonr(nat_track, obs_track)[0]

                safe_name = region["name"].replace("/", "_").replace(" ", "_")
                plot_region_tracks(
                    ft_track, obs_track, region["name"],
                    regions_dir / f"{safe_name}_{res}bp.png",
                    res=res, native_pred=nat_track,
                    ft_r=ft_r, native_r=nat_r,
                )

        log.info("Region plots saved to %s", regions_dir)

    # ---- Feature: ISM ----
    if run_ism:
        log.info("=" * 60)
        log.info("In-silico mutagenesis")
        log.info("=" * 60)

        ism_dir = out_dir / "ism"
        regions = parse_regions_bed(args.regions_bed)

        run_ism_for_regions(
            model, args.genome, regions, modality,
            args.ism_window_size, device, ism_dir,
        )

    # ---- Summary ----
    summary_text = format_summary_table(
        ft_metrics_by_res if ft_metrics_by_res else None,
        native_metrics_by_res if native_metrics_by_res else None,
        native_display_name,
        resolutions,
    )
    if summary_text:
        print(summary_text)
        with open(out_dir / "summary.txt", "w") as f:
            f.write(summary_text)

    # JSON summary
    native_info = None
    if native_display_name is not None:
        native_info = {
            "biosample": native_display_name,
            "track_index": native_track_idx,
        }

    save_summary_json(
        ft_metrics_by_res if ft_metrics_by_res else None,
        native_metrics_by_res if native_metrics_by_res else None,
        {
            k: v for k, v in ckpt_meta.items()
            if not isinstance(v, (list, dict)) or k in ("modality", "track_names")
        },
        native_info, loss,
        out_dir / "summary.json",
    )

    log.info("Done. Output: %s", out_dir)
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            log.info("  %s", p.relative_to(out_dir))


if __name__ == "__main__":
    main()

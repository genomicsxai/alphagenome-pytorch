#!/usr/bin/env python
"""Evaluate an AlphaGenome checkpoint on exon-specific metrics.

RNA profile Pearson follows the AlphaGenome paper by applying log(1 + x).
Jensen-Shannon metrics use raw, non-negative profiles. Regional total-count
metrics are computed from the cropped 1-bp arrays and are therefore identical
for every reported display resolution.

Example:

    python scripts/compute_gene_metrics.py \
        --checkpoint best_model.pth \
        --pretrained-weights model_fold1.pth \
        --genome GRCh38.fa \
        --bigwig sample_forward.bw sample_reverse.bw \
        --test-bed fold_1/test.bed \
        --gtf-parquet gencode.v46.annotation.gtf.parquet \
        --samples sample \
        --sequence-length 1048576 \
        --score-window-bp 196608 \
        --gene-score-window-bp 196608 \
        --bin-sizes 1,32 \
        --output-dir evaluation/rna_seq
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader, Subset

from alphagenome_pytorch.extensions.finetuning.datasets import GenomicDataset
from alphagenome_pytorch.extensions.finetuning.evaluation import evaluate_split
from alphagenome_pytorch.extensions.finetuning.training import collate_genomic

from evaluate_checkpoint import (
    build_metric_views,
    center_crop_profiles,
    compute_comparison_metrics,
    load_finetuned_model,
    _pearson_or_nan,
    parse_metric_bin_sizes,
)


log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an AlphaGenome checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained-weights", required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument(
        "--bigwig", nargs="+", required=True,
        help="RNA-seq coverage BigWigs in checkpoint track order",
    )
    parser.add_argument("--test-bed", required=True, help="fold_1 test BED")
    parser.add_argument(
        "--gtf-parquet", required=True,
        help="Processed GTF parquet containing exon coordinates and gene metadata",
    )
    parser.add_argument(
        "--samples", nargs="+", default=None,
        help=(
            "Sample names for interleaved forward/reverse track pairs. If omitted, "
            "sample_0, sample_1, ... are used."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=131_072)
    parser.add_argument(
        "--score-window-bp", type=int, default=None,
        help="Centered post-inference width for regional profile metrics",
    )
    parser.add_argument(
        "--gene-score-window-bp", type=int, default=None,
        help=(
            "Centered post-inference width for TSS selection and gene exon-mean "
            "metrics; omit to use the full inference interval"
        ),
    )
    parser.add_argument("--source-resolution", type=int, default=1)
    parser.add_argument("--bin-sizes", default="1,32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-regions", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def _clean_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    return value


def load_gtf(gtf_parquet: str) -> pd.DataFrame:
    """Load columns needed by the official AlphaGenome gene-mask path."""
    return pd.read_parquet(
        gtf_parquet,
        columns=[
            "Chromosome", "Start", "End", "Strand", "Feature",
            "gene_id", "gene_name", "gene_type", "transcript_id",
        ],
    )


# Direct NumPy ports of the public AlphaGenome gene-mask implementation in
# alphagenome/data/gene_annotation.py and
# alphagenome_research/model/variant_scoring/gene_mask_extractor.py.
def extract_tss(gtf: pd.DataFrame, feature: str = "transcript") -> pd.DataFrame:
    """Extract strand-aware transcription start sites from a GTF DataFrame."""
    tss = gtf[gtf.Feature == feature].copy()
    tss["feature_start"] = tss.Start
    tss["feature_end"] = tss.End
    new_start = np.where(tss.Strand == "-", tss.End, tss.Start)
    tss.Start = new_start
    tss.End = new_start
    return tss


class _PositionExtractor:
    """Extract rows whose position lies in a half-open genomic interval."""

    def __init__(
        self,
        frame: pd.DataFrame,
        position_column: str,
        chromosome_column: str = "Chromosome",
    ):
        self._df_position = {
            chromosome: (group, group[position_column].values)
            for chromosome, group in frame.groupby(
                chromosome_column, observed=False,
            )
        }
        self._df_empty = frame.iloc[:0]

    def extract(self, chromosome: str, start: int, end: int) -> pd.DataFrame:
        if chromosome not in self._df_position:
            return self._df_empty
        frame, position = self._df_position[chromosome]
        return frame[(position >= start) & (position < end)]


class _ExonExtractor:
    """Extract exon intervals for one transcript."""

    def __init__(self, gtf: pd.DataFrame):
        self._exons_by_transcript_id = gtf[gtf.Feature == "exon"][
            ["Chromosome", "Start", "End", "Strand", "transcript_id"]
        ].groupby("transcript_id", sort=False)

    def extract(self, transcript_id: str) -> list[tuple[str, int, int, str]]:
        try:
            exons = self._exons_by_transcript_id.get_group(transcript_id)
        except KeyError:
            return []
        return list(zip(
            exons.Chromosome,
            exons.Start.astype(int),
            exons.End.astype(int),
            exons.Strand,
        ))


class _ExonMaskExtractor:
    """Generate a boolean exon mask for one transcript."""

    def __init__(self, gtf: pd.DataFrame):
        self._exon_extractor = _ExonExtractor(gtf)

    def extract(
        self,
        chromosome: str,
        start: int,
        end: int,
        transcript_id: str,
    ) -> np.ndarray:
        mask = np.zeros(end - start, dtype=bool)
        for exon_chromosome, exon_start, exon_end, _ in (
            self._exon_extractor.extract(transcript_id)
        ):
            if exon_chromosome != chromosome:
                continue
            if exon_start < end and exon_end > start:
                relative_start = max(exon_start - start, 0)
                relative_end = min(exon_end - start, end - start)
                mask[relative_start:relative_end] = True
        return mask


class _OfficialGeneExonMaskExtractor:
    """Port of GeneMaskExtractor(EXONS, INTERVAL_CONTAINED)."""

    _GENE_COLUMNS = [
        "Chromosome", "Start", "End", "Strand", "gene_id", "gene_name",
        "gene_type",
    ]

    def __init__(self, gtf: pd.DataFrame):
        self._exon_mask_extractor = _ExonMaskExtractor(gtf)
        self._tss = _PositionExtractor(
            extract_tss(gtf), position_column="Start",
        )
        self._gene_df = gtf[gtf.Feature == "gene"][self._GENE_COLUMNS].set_index(
            "gene_id"
        )

    def extract(
        self, chromosome: str, start: int, end: int,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        transcript_subset = self._tss.extract(chromosome, start, end)
        gene_masks: dict[str, np.ndarray] = {}

        for row in transcript_subset.itertuples():
            exon_mask = self._exon_mask_extractor.extract(
                chromosome, start, end, row.transcript_id,
            )
            if (gene_mask := gene_masks.get(row.gene_id)) is not None:
                gene_mask |= exon_mask
            else:
                gene_masks[row.gene_id] = exon_mask

        unique_gene_ids = list(transcript_subset["gene_id"].unique())
        if not unique_gene_ids:
            return (
                np.empty((end - start, 0), dtype=bool),
                pd.DataFrame(columns=[
                    "gene_id", "strand", "gene_name", "gene_type",
                    "interval_start", "Chromosome", "Start", "End",
                ]),
            )
        gene_metadata = self._gene_df.loc[unique_gene_ids]
        mask = np.empty((end - start, len(unique_gene_ids)), dtype=bool)
        for index, gene_id in enumerate(unique_gene_ids):
            mask[:, index] = gene_masks[gene_id]

        metadata = pd.DataFrame({
            "gene_id": unique_gene_ids,
            "strand": gene_metadata.Strand.values,
            "gene_name": gene_metadata.gene_name.values,
            "gene_type": gene_metadata.gene_type.values,
            "interval_start": [start] * len(unique_gene_ids),
            "Chromosome": gene_metadata.Chromosome.values,
            "Start": gene_metadata.Start.values,
            "End": gene_metadata.End.values,
        }).reset_index(drop=True)
        return mask, metadata


def downsample_gene_mask(mask: np.ndarray, resolution: int) -> np.ndarray:
    """Match GeneIntervalScorer's max pooling of masks at coarse resolution."""
    if resolution == 1:
        return mask
    if mask.shape[0] % resolution:
        raise ValueError(
            f"Gene mask length {mask.shape[0]} is not divisible by {resolution}"
        )
    return mask.reshape(
        mask.shape[0] // resolution, resolution, -1,
    ).max(axis=1)


def score_gene_interval_mean(tracks: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Direct NumPy port of GeneIntervalScorer.MEAN."""
    return np.einsum("lt,lg->gt", tracks, masks) / np.expand_dims(
        masks.sum(axis=0), axis=-1,
    )


def assign_genes_to_nearest_center(
    gtf: pd.DataFrame,
    positions: list[tuple[str, int, int]],
) -> dict[str, int]:
    """Assign each TSS-selected gene to one overlapping evaluation interval.

    AlphaGenome's extractor scores genes independently in every requested
    interval. The fold-1 BED contains overlapping windows, so a dataset-level
    correlation additionally needs a deterministic one-row-per-gene rule.
    Among intervals containing a transcript TSS, use the interval whose center
    is closest to any TSS for that gene (ties resolve to the earlier BED row).
    """
    tss_extractor = _PositionExtractor(
        extract_tss(gtf), position_column="Start",
    )
    best: dict[str, tuple[float, int]] = {}
    for interval_index, (chromosome, start, end) in enumerate(positions):
        transcript_subset = tss_extractor.extract(chromosome, start, end)
        if transcript_subset.empty:
            continue
        center = (start + end) / 2
        distances = (transcript_subset["Start"] - center).abs()
        for gene_id, distance in distances.groupby(
            transcript_subset["gene_id"], sort=False,
        ).min().items():
            candidate = (float(distance), interval_index)
            if gene_id not in best or candidate < best[gene_id]:
                best[gene_id] = candidate
    return {gene_id: interval_index for gene_id, (_, interval_index) in best.items()}


def center_crop_genomic_positions(
    positions: list[tuple[str, int, int]],
    score_window_bp: int | None,
    source_resolution: int = 1,
) -> list[tuple[str, int, int]]:
    """Return genomic coordinates matching a centered profile-array crop."""
    if score_window_bp is None:
        return positions
    if (
        score_window_bp <= 0
        or source_resolution <= 0
        or score_window_bp % source_resolution
    ):
        raise ValueError(
            "gene_score_window_bp must be positive and divisible by "
            "source_resolution"
        )

    cropped = []
    for chromosome, start, end in positions:
        available_bp = end - start
        if score_window_bp > available_bp:
            raise ValueError(
                f"Requested {score_window_bp:,} bp gene scoring window, but "
                f"interval {chromosome}:{start}-{end} spans only "
                f"{available_bp:,} bp"
            )
        if available_bp % source_resolution:
            raise ValueError(
                f"Interval width {available_bp} is not divisible by source "
                f"resolution {source_resolution}"
            )
        available_bins = available_bp // source_resolution
        score_bins = score_window_bp // source_resolution
        left_bins = (available_bins - score_bins) // 2
        crop_start = start + left_bins * source_resolution
        cropped.append((chromosome, crop_start, crop_start + score_window_bp))
    return cropped


def align_test_intervals_to_dataset(
    test_bed: str,
    dataset: GenomicDataset,
) -> pd.DataFrame:
    """Match original BED intervals to the windows retained by GenomicDataset."""
    intervals = pd.read_csv(
        test_bed,
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
    )
    retained: list[dict] = []
    position_index = 0
    half_length = dataset.sequence_length // 2

    for original_index, row in intervals.iterrows():
        chrom = row["chrom"]
        start, end = int(row["start"]), int(row["end"])
        if chrom not in dataset._chrom_sizes:
            continue
        if end - start == dataset.sequence_length:
            window_start, window_end = start, end
        else:
            center = (start + end) // 2
            window_start = center - half_length
            window_end = window_start + dataset.sequence_length
        if window_start < 0 or window_end > dataset._chrom_sizes[chrom]:
            continue

        expected = (chrom, window_start, window_end)
        if position_index >= len(dataset._positions_list):
            raise ValueError("More valid BED intervals than GenomicDataset positions")
        actual = dataset._positions_list[position_index]
        if expected != actual:
            raise ValueError(
                f"BED/dataset interval mismatch at retained index {position_index}: "
                f"expected {expected}, found {actual}"
            )
        retained.append({
            "original_interval_idx": int(original_index),
            "chrom": chrom,
            "start": start,
            "end": end,
            "window_start": window_start,
            "window_end": window_end,
        })
        position_index += 1

    if position_index != len(dataset._positions_list):
        raise ValueError(
            f"Matched {position_index} intervals but dataset retained "
            f"{len(dataset._positions_list)}"
        )
    return pd.DataFrame(retained)


def collect_gene_expression_rows(
    gtf: pd.DataFrame,
    test_intervals: pd.DataFrame,
    positions: list[tuple[str, int, int]],
    prediction_views: dict[int, np.ndarray],
    target_views: dict[int, np.ndarray],
    bin_sizes: tuple[int, ...],
    sample_names: list[str],
) -> pd.DataFrame:
    """Collect one official exon-mean expression row per gene and track."""
    columns = [
        "bin_size_bp", "gene_id", "gene_name", "chrom", "strand",
        "interval_index", "original_interval_index", "track_index", "sample",
        "pred_mean", "obs_mean", "pred_log1p_mean", "obs_log1p_mean",
    ]
    rows: list[dict] = []
    gene_owner = assign_genes_to_nearest_center(gtf, positions)
    mask_extractor = _OfficialGeneExonMaskExtractor(gtf)
    for interval_index, (chrom, window_start, window_end) in enumerate(positions):
        one_bp_masks, metadata = mask_extractor.extract(
            chrom, window_start, window_end,
        )
        keep = metadata["gene_id"].map(gene_owner).eq(interval_index).to_numpy()
        metadata = metadata.loc[keep].reset_index(drop=True)
        one_bp_masks = one_bp_masks[:, keep]
        if metadata.empty:
            continue
        for bin_size in bin_sizes:
            masks = downsample_gene_mask(one_bp_masks, bin_size)
            pred_means = score_gene_interval_mean(
                prediction_views[bin_size][interval_index], masks,
            )
            target_means = score_gene_interval_mean(
                target_views[bin_size][interval_index], masks,
            )
            for gene_index, gene in metadata.iterrows():
                strand = gene["strand"]
                track_indices = range(
                    0 if strand == "+" else 1, len(sample_names) * 2, 2,
                )
                for track_index in track_indices:
                    if track_index >= pred_means.shape[1]:
                        continue
                    rows.append({
                        "bin_size_bp": bin_size,
                        "gene_id": gene["gene_id"],
                        "gene_name": gene["gene_name"],
                        "chrom": chrom,
                        "strand": strand,
                        "interval_index": interval_index,
                        "original_interval_index": int(
                            test_intervals.iloc[interval_index][
                                "original_interval_idx"
                            ]
                        ),
                        "track_index": track_index,
                        "sample": sample_names[track_index // 2],
                        "pred_mean": float(pred_means[gene_index, track_index]),
                        "obs_mean": float(target_means[gene_index, track_index]),
                        "pred_log1p_mean": float(np.log1p(
                            pred_means[gene_index, track_index]
                        )),
                        "obs_log1p_mean": float(np.log1p(
                            target_means[gene_index, track_index]
                        )),
                    })
    return pd.DataFrame(rows, columns=columns)


def summarize_gene_expression(
    rows: pd.DataFrame,
    bin_sizes: tuple[int, ...],
) -> tuple[dict[int, dict], pd.DataFrame]:
    """Summarize log1p exon-mean expression across gene-track pairs."""
    summary: dict[int, dict] = {}
    per_track_rows: list[dict] = []
    for bin_size in bin_sizes:
        subset = rows[rows["bin_size_bp"] == bin_size]
        pred = subset["pred_log1p_mean"].to_numpy(dtype=np.float64)
        obs = subset["obs_log1p_mean"].to_numpy(dtype=np.float64)
        summary[bin_size] = {
            "pearson_r_log1p_exon_mean": _pearson_or_nan(pred, obs),
            "spearman_r_log1p_exon_mean": float(stats.spearmanr(pred, obs)[0])
            if len(pred) >= 2 else float("nan"),
            "mse_log1p_exon_mean": float(np.mean(np.square(pred - obs)))
            if len(pred) else float("nan"),
            "mae_log1p_exon_mean": float(np.mean(np.abs(pred - obs)))
            if len(pred) else float("nan"),
            "n_gene_track_pairs": int(len(subset)),
            "n_unique_genes": int(subset["gene_id"].nunique()),
        }
        track_pearsons: list[float] = []
        for (track_index, sample), track_rows in subset.groupby(
            ["track_index", "sample"], sort=True,
        ):
            track_pred = track_rows["pred_log1p_mean"].to_numpy(dtype=np.float64)
            track_obs = track_rows["obs_log1p_mean"].to_numpy(dtype=np.float64)
            track_pearson = _pearson_or_nan(track_pred, track_obs)
            track_pearsons.append(track_pearson)
            per_track_rows.append({
                "bin_size_bp": bin_size,
                "track_index": int(track_index),
                "sample": sample,
                "pearson_r_log1p_exon_mean": track_pearson,
                "n_genes": len(track_rows),
            })
        finite_track_pearsons = np.asarray(track_pearsons, dtype=np.float64)
        finite_track_pearsons = finite_track_pearsons[
            np.isfinite(finite_track_pearsons)
        ]
        summary[bin_size]["pearson_r_mean_across_tracks"] = (
            float(finite_track_pearsons.mean())
            if finite_track_pearsons.size else float("nan")
        )
        summary[bin_size]["n_tracks_with_valid_pearson"] = int(
            finite_track_pearsons.size
        )
    return summary, pd.DataFrame(per_track_rows)


def _write_per_region_csv(
    path: Path,
    positions: list[tuple[str, int, int]],
    metrics_by_bin: dict[int, dict],
    pred_totals: np.ndarray,
    target_totals: np.ndarray,
) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bin_size_bp", "region_index", "chrom", "start", "end",
                "profile_pearson_r", "js_divergence", "js_distance",
                "pred_total", "target_total",
            ],
        )
        writer.writeheader()
        for bin_size, metrics in metrics_by_bin.items():
            for index, (chrom, start, end) in enumerate(positions):
                writer.writerow({
                    "bin_size_bp": bin_size,
                    "region_index": index,
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "profile_pearson_r": metrics["profile_pearson_r_all"][index],
                    "js_divergence": metrics["js_divergence_all"][index],
                    "js_distance": metrics["js_distance_all"][index],
                    # One total per region is expected for the usual single
                    # RNA coverage track. Multiple tracks are summed here and
                    # remain available individually in saved predictions.
                    "pred_total": pred_totals[index].sum(),
                    "target_total": target_totals[index].sum(),
                })


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    try:
        bin_sizes = parse_metric_bin_sizes(args.bin_sizes)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")
    assert bin_sizes is not None

    if args.score_window_bp is not None and args.score_window_bp > args.sequence_length:
        sys.exit(
            "Error: --score-window-bp cannot exceed --sequence-length. "
            "Do not request the Borzoi window for a 131-kbp checkpoint."
        )
    if (
        args.gene_score_window_bp is not None
        and args.gene_score_window_bp > args.sequence_length
    ):
        sys.exit(
            "Error: --gene-score-window-bp cannot exceed --sequence-length."
        )

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint_meta = load_finetuned_model(
        args.checkpoint, args.pretrained_weights, device,
    )
    modality = checkpoint_meta["modality"]
    if isinstance(modality, list):
        if len(modality) != 1:
            sys.exit(f"Error: expected one RNA modality, found {modality}")
        modality = modality[0]
    if modality not in {"rna_seq", "rna-seq", "rnaseq"}:
        sys.exit(f"Error: checkpoint modality is {modality!r}, not RNA-seq")

    resolutions = checkpoint_meta["resolutions"]
    if isinstance(resolutions, dict):
        resolutions = resolutions.get(modality, (args.source_resolution,))
    resolutions = tuple(resolutions)
    if args.source_resolution not in resolutions:
        sys.exit(
            f"Error: source resolution {args.source_resolution} is not in "
            f"checkpoint resolutions {resolutions}"
        )

    dataset = GenomicDataset(
        genome_fasta=args.genome,
        bigwig_files=args.bigwig,
        bed_file=args.test_bed,
        resolutions=resolutions,
        sequence_length=args.sequence_length,
    )
    test_intervals = align_test_intervals_to_dataset(args.test_bed, dataset)
    if args.max_regions > 0 and len(dataset) > args.max_regions:
        rng = np.random.default_rng(42)
        indices = rng.choice(
            len(dataset), args.max_regions, replace=False,
        ).tolist()
        positions = [dataset._positions_list[index] for index in indices]
        test_intervals = test_intervals.iloc[indices].reset_index(drop=True)
        evaluation_dataset = Subset(dataset, indices)
    else:
        positions = dataset._positions_list
        test_intervals = test_intervals.reset_index(drop=True)
        evaluation_dataset = dataset

    if len(args.bigwig) % 2:
        sys.exit(
            "Error: gene-expression evaluation expects interleaved "
            "forward/reverse BigWig track pairs"
        )
    n_samples = len(args.bigwig) // 2
    sample_names = args.samples or [f"sample_{index}" for index in range(n_samples)]
    if len(sample_names) != n_samples:
        sys.exit(
            f"Error: expected {n_samples} --samples values for {len(args.bigwig)} "
            f"interleaved tracks, received {len(sample_names)}"
        )

    loader = DataLoader(
        evaluation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_genomic,
    )
    predictions, targets, loss = evaluate_split(
        model, modality, loader, device, resolutions,
    )
    source = args.source_resolution
    cropped_predictions, cropped_targets = center_crop_profiles(
        predictions[source], targets[source], args.score_window_bp, source,
    )
    prediction_views, target_views = build_metric_views(
        predictions[source],
        targets[source],
        source_resolution=source,
        bin_sizes=bin_sizes,
        score_window_bp=args.score_window_bp,
        reduction="mean",
    )
    gene_prediction_views, gene_target_views = build_metric_views(
        predictions[source],
        targets[source],
        source_resolution=source,
        bin_sizes=bin_sizes,
        score_window_bp=args.gene_score_window_bp,
        reduction="mean",
    )
    gene_positions = center_crop_genomic_positions(
        positions, args.gene_score_window_bp, source,
    )

    # Regional counts are defined from the cropped 1-bp signal, not from the
    # mean-pooled display profiles.
    count_metrics = compute_comparison_metrics(
        cropped_predictions, cropped_targets, profile_transform="log1p",
    )
    pred_totals = cropped_predictions.sum(axis=1)
    target_totals = cropped_targets.sum(axis=1)

    metrics_by_bin: dict[int, dict] = {}
    for bin_size in bin_sizes:
        metrics = compute_comparison_metrics(
            prediction_views[bin_size],
            target_views[bin_size],
            profile_transform="log1p",
        )
        for key in ("count_pearson_r", "count_pearson_r_raw", "count_pearson_r_log1p"):
            metrics[key] = count_metrics[key]
        metrics_by_bin[bin_size] = metrics
        log.info(
            "%dbp: profile r=%.4f, accumulated r=%.4f, log-count r=%.4f, "
            "JS distance=%.4f",
            bin_size,
            metrics["profile_pearson_r_mean"],
            metrics["track_pearson_r_accumulated_mean"],
            metrics["count_pearson_r_log1p"],
            metrics["js_distance_mean"],
        )

    gtf = load_gtf(args.gtf_parquet)
    gene_rows = collect_gene_expression_rows(
        gtf=gtf,
        test_intervals=test_intervals,
        positions=gene_positions,
        prediction_views=gene_prediction_views,
        target_views=gene_target_views,
        bin_sizes=bin_sizes,
        sample_names=sample_names,
    )
    gene_metrics, gene_metrics_per_track = summarize_gene_expression(
        gene_rows, bin_sizes,
    )
    for bin_size, metrics in gene_metrics.items():
        log.info(
            "%dbp gene expression: Pearson r=%.4f across %d genes (%d pairs)",
            bin_size,
            metrics["pearson_r_log1p_exon_mean"],
            metrics["n_unique_genes"],
            metrics["n_gene_track_pairs"],
        )

    summary = {
        "checkpoint": checkpoint_meta,
        "evaluation": {
            "modality": modality,
            "test_bed": args.test_bed,
            "fold": "fold_1 expected; determined by --test-bed",
            "sequence_length_bp": args.sequence_length,
            "score_window_bp": args.score_window_bp or args.sequence_length,
            "source_resolution_bp": source,
            "bin_sizes_bp": list(bin_sizes),
            "profile_transform": "log1p",
            "profile_pooling": "mean",
            "count_source": "cropped_source_resolution_profile",
            "gene_score_window_bp": (
                args.gene_score_window_bp or args.sequence_length
            ),
            "gene_expression_window": (
                "center_crop" if args.gene_score_window_bp is not None
                else "full_inference_test_interval"
            ),
            "gene_selection": "official_transcript_tss",
            "gene_interval_assignment": "nearest_interval_center_to_gene_tss",
            "gene_mask_aggregation": "union_exon_mask_then_mean",
            "gene_expression_transform": "log1p_mean_exonic_coverage",
            "gtf_parquet": args.gtf_parquet,
            "n_regions": len(positions),
            "loss": loss,
        },
        "metrics": metrics_by_bin,
        "gene_expression_metrics": gene_metrics,
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(_clean_json(summary), handle, indent=2, default=str)
    _write_per_region_csv(
        output_dir / "metrics_per_region.csv",
        positions,
        metrics_by_bin,
        pred_totals,
        target_totals,
    )
    gene_rows.to_csv(output_dir / "gene_expression_per_gene.csv", index=False)
    gene_metrics_per_track.to_csv(
        output_dir / "gene_expression_metrics_per_track.csv", index=False,
    )

    if args.save_predictions:
        prediction_dir = output_dir / "predictions"
        prediction_dir.mkdir(exist_ok=True)
        np.save(
            prediction_dir / "predictions_scored_1bp.npy",
            cropped_predictions.astype(np.float16),
        )
        np.save(
            prediction_dir / "targets_scored_1bp.npy",
            cropped_targets.astype(np.float16),
        )
        for bin_size in bin_sizes:
            if bin_size == source:
                continue
            np.save(
                prediction_dir / f"predictions_scored_{bin_size}bp.npy",
                prediction_views[bin_size].astype(np.float16),
            )
            np.save(
                prediction_dir / f"targets_scored_{bin_size}bp.npy",
                target_views[bin_size].astype(np.float16),
            )

    log.info("Wrote RNA-seq evaluation to %s", output_dir)


if __name__ == "__main__":
    main()

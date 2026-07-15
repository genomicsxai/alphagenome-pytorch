"""Interval / gene aggregation for AlphaGenome-PyTorch predictions.

This module turns per-position predictions (``[B, S, C]``) into per-region
matrices — most usefully **per-gene × per-track** counts and expression values —
using a single positional primitive that all consumers share.

Two user-facing helpers build on the primitive:

* :func:`aggregate_genes` — loss-style **gene-body** counts (exons + introns),
  mirroring the training gene-LFC aggregation. Linear space.
* :func:`gene_expression` — gene-expression:
  **log-transformed mean coverage over a gene's annotated exons**, strand-matched,
  keeping genes with ≥50% of their exons inside the interval.

Both return a :class:`GeneCounts` object with torch-native data plus DataFrame /
AnnData converters. The correlation helpers (:func:`normalize_expression`,
:func:`gene_expression_correlations`) implement the three gene-expression
correlation flavors and are reused by the fine-tuning validation metric.

Design notes
------------
* The core (``aggregate_intervals`` + the correlation helpers) is **pure tensor**
  code — no pandas / anndata — so the training validation loop can import it
  without pulling optional deps.
* Aggregation is **always purely positional**. ``strand`` is post-processing on
  the resulting ``[genes × tracks]`` matrix (see :func:`aggregate_genes`).
* Users bring their own GTF/parquet; no bundled annotation. ``anndata`` is an
  optional, lazily-imported dependency.

Units
-----
AlphaGenome's 128bp predictions are **bin sums**, not per-base values —
``heads.predictions_scaling`` multiplies by ``resolution`` to reach experimental
space, and the fine-tuning pipeline bins targets the same way. So a mean over a
region must divide by *bases*, not by elements, or it comes out ``resolution``×
too large. That is what ``bin_size`` is for.

This buys **unit consistency**, not exact agreement between resolutions. Region
means at 128bp remain an approximation: :meth:`GeneAnnotation.get_exon_bin_ranges`
includes every bin an exon *touches*, and such a boundary bin is an already-summed
total that mixes exonic and non-exonic bases. No choice of divisor can separate
them after the fact. Only regions whose boundaries land on bin edges agree exactly
with the 1bp result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import torch
from torch import Tensor

from .metrics import pearson_r

if TYPE_CHECKING:  # avoid importing heavy / optional modules at import time
    import pandas as pd
    from .named_outputs import TrackMetadata
    from .variant_scoring.annotations import GeneAnnotation


__all__ = [
    "aggregate_intervals",
    "normalize_expression",
    "gene_expression_correlations",
    "GeneCounts",
    "GeneCountAccumulator",
    "aggregate_genes",
    "gene_expression",
    "gene_expression_values",
    "combine_gene_expression",
]


# --------------------------------------------------------------------------- #
# Core primitive
# --------------------------------------------------------------------------- #
def aggregate_intervals(
    predictions: Tensor,
    mask: Tensor,
    reduce: str = "sum",
    bin_size: int = 1,
) -> Tensor:
    """Aggregate per-position predictions over R interval masks.

    Args:
        predictions: ``[B, S, C]`` (or ``[S, C]``) per-element values.
        mask: ``[S, R]`` interval masks (bool or float); column ``r`` selects the
            elements belonging to region ``r``.
        reduce: ``"sum"`` for raw summed signal, or ``"mean"`` for the
            length-normalized mean (divides by each region's masked length,
            clamped to ``>= 1`` so an empty mask yields 0 rather than NaN).
        bin_size: Number of bases integrated into each element of
            ``predictions``. Only ``"mean"`` uses it: the divisor becomes
            ``masked_elements * bin_size``, i.e. bases rather than elements.
            Leave at 1 for per-base inputs (1bp predictions). Pass the bin width
            only when the inputs are **bin sums** — AlphaGenome's 128bp outputs
            are, since ``predictions_scaling`` multiplies by ``resolution`` to
            reach experimental space. Passing it for per-base inputs would
            wrongly shrink the mean by that factor.

    Returns:
        ``[B, R, C]`` aggregated values (a leading batch axis is always present,
        even when ``predictions`` was 2-D).
    """
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if predictions.dim() != 3:
        raise ValueError(
            f"predictions must be [B, S, C] or [S, C], got shape {tuple(predictions.shape)}"
        )
    if mask.dim() != 2:
        raise ValueError(f"mask must be [S, R], got shape {tuple(mask.shape)}")
    if mask.shape[0] != predictions.shape[1]:
        raise ValueError(
            f"mask length {mask.shape[0]} != predictions sequence length "
            f"{predictions.shape[1]}"
        )
    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}")

    mask_f = mask.to(predictions.dtype)
    out = torch.einsum("bsc,sr->brc", predictions, mask_f)  # [B, R, C]

    if reduce == "sum":
        return out
    if reduce == "mean":
        lengths = (mask_f.sum(dim=0) * bin_size).clamp(min=1.0)  # [R], in bases
        return out / lengths[None, :, None]
    raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}")


# --------------------------------------------------------------------------- #
# Correlation helpers (gene-expression)
# --------------------------------------------------------------------------- #
def _nan_pearson(pred: Tensor, obs: Tensor, dim: int, eps: float = 1e-8) -> Tensor:
    """Pearson correlation over ``dim``, ignoring entries where either is NaN.

    Positions that are NaN in ``pred`` or ``obs`` are dropped pairwise (their
    contribution is zeroed and excluded from the counts / means). Columns with
    fewer than two valid points return NaN.
    """
    pred = pred.float()
    obs = obs.float()
    valid = ~(torch.isnan(pred) | torch.isnan(obs))
    w = valid.float()

    n = w.sum(dim=dim, keepdim=True)  # valid count along dim
    pred0 = torch.where(valid, pred, torch.zeros_like(pred))
    obs0 = torch.where(valid, obs, torch.zeros_like(obs))

    mean_p = (pred0.sum(dim=dim, keepdim=True)) / n.clamp(min=1)
    mean_o = (obs0.sum(dim=dim, keepdim=True)) / n.clamp(min=1)

    pc = torch.where(valid, pred - mean_p, torch.zeros_like(pred))
    oc = torch.where(valid, obs - mean_o, torch.zeros_like(obs))

    num = (pc * oc).sum(dim=dim)
    den = pc.pow(2).sum(dim=dim).sqrt() * oc.pow(2).sum(dim=dim).sqrt()
    r = num / (den + eps)

    # Undefined where fewer than 2 valid points.
    enough = (n.squeeze(dim) >= 2)
    return torch.where(enough, r, torch.full_like(r, float("nan")))


def normalize_expression(matrix: Tensor) -> Tensor:
    """Quantile-normalize across genes per track, then gene-mean-center.

    Implements the AlphaGenome *specificity* normalization: for each track
    (column) the gene values are quantile-normalized across genes to a common
    reference (the mean of the per-column sorted values); then each gene's (row)
    mean across tracks is subtracted.

    Args:
        matrix: ``[G, C]`` gene × track expression (no NaN). Genes are rows.

    Returns:
        ``[G, C]`` normalized matrix.
    """
    if matrix.dim() != 2:
        raise ValueError(f"matrix must be [G, C], got {tuple(matrix.shape)}")
    x = matrix.float()
    g, c = x.shape
    if g < 2 or c < 1:
        return x - x.mean(dim=1, keepdim=True) if g >= 1 and c >= 1 else x

    # Quantile normalization per column (track), across genes (rows).
    order = x.argsort(dim=0)                    # [G, C] indices sorting each column
    ranks = order.argsort(dim=0)               # [G, C] rank of each entry in its column
    sorted_x, _ = x.sort(dim=0)                # [G, C] sorted values per column
    reference = sorted_x.mean(dim=1)           # [G] mean of the sorted values across tracks
    qn = reference[ranks]                      # map each entry to its rank's reference value

    # Gene-mean centering (subtract each gene's mean across tracks).
    qn = qn - qn.mean(dim=1, keepdim=True)
    return qn


def gene_expression_correlations(
    pred: Tensor,
    obs: Tensor,
    *,
    eps: float = 1e-8,
) -> dict[str, float]:
    """The three AlphaGenome gene-expression correlations.

    Args:
        pred: ``[G, C]`` predicted (log-space) gene × track expression.
        obs: ``[G, C]`` observed (log-space) gene × track expression. NaN cells
            (e.g. strand-incompatible) are handled pairwise.
        eps: numerical epsilon.

    Returns:
        Dict with mean correlations:
          * ``across_genes`` — raw, per-track across genes
          * ``across_genes_norm`` — quantile-normalized + gene-mean-centered,
            per-track across genes
          * ``across_tracks_norm`` — same normalized data, per-gene across tracks
    """
    if pred.shape != obs.shape:
        raise ValueError(f"pred {tuple(pred.shape)} and obs {tuple(obs.shape)} must match")

    # Raw / across genes: correlate over genes (dim=0), one r per track, mean.
    raw = _nan_pearson(pred, obs, dim=0, eps=eps)  # [C]

    result = {"across_genes": float(_nanmean(raw))}

    # Normalized variants need a dense matrix; drop genes/tracks that are all-NaN.
    pred_d, obs_d = _dense_common(pred, obs)
    if pred_d is not None and pred_d.shape[0] >= 2 and pred_d.shape[1] >= 2:
        pred_n = normalize_expression(pred_d)
        obs_n = normalize_expression(obs_d)
        ag_norm = _nan_pearson(pred_n, obs_n, dim=0, eps=eps)   # [C]
        at_norm = _nan_pearson(pred_n, obs_n, dim=1, eps=eps)   # [G]
        result["across_genes_norm"] = float(_nanmean(ag_norm))
        result["across_tracks_norm"] = float(_nanmean(at_norm))
    else:
        result["across_genes_norm"] = float("nan")
        result["across_tracks_norm"] = float("nan")
    return result


def _nanmean(x: Tensor) -> Tensor:
    valid = ~torch.isnan(x)
    if valid.sum() == 0:
        return torch.tensor(float("nan"))
    return x[valid].mean()


def _dense_common(pred: Tensor, obs: Tensor):
    """Drop rows/cols that are entirely NaN in either matrix (for QN)."""
    finite = ~(torch.isnan(pred) | torch.isnan(obs))
    keep_g = finite.any(dim=1)
    keep_c = finite.any(dim=0)
    if keep_g.sum() < 2 or keep_c.sum() < 2:
        return None, None
    pred_d = pred[keep_g][:, keep_c]
    obs_d = obs[keep_g][:, keep_c]
    # Remaining NaNs (sparse strand mismatches) → 0 after centering is imperfect;
    # replace with column means so QN ranks are well-defined.
    pred_d = _fill_nan_colmean(pred_d)
    obs_d = _fill_nan_colmean(obs_d)
    return pred_d, obs_d


def _fill_nan_colmean(x: Tensor) -> Tensor:
    nan = torch.isnan(x)
    if not nan.any():
        return x
    x = x.clone()
    # column means over non-NaN entries
    valid = (~nan).float()
    means = (torch.where(nan, torch.zeros_like(x), x)).sum(0) / valid.sum(0).clamp(min=1)
    x[nan] = means.repeat(x.shape[0], 1)[nan]
    return x


# --------------------------------------------------------------------------- #
# Result object
# --------------------------------------------------------------------------- #
@dataclass
class GeneCounts:
    """Per-gene × per-track aggregated result with export converters.

    Attributes:
        counts: ``[B, G, C]`` aggregated values (``space`` labels the scale).
        gene_metadata: ``G``-row DataFrame (→ AnnData ``var``): gene_id,
            gene_name, gene_type, strand, Start, End.
        track_metadata: ``C``-row DataFrame (→ AnnData ``obs``): track_index,
            track_name, strand, and any extras.
        space: ``"linear"`` or ``"log"``.
    """

    counts: Tensor
    gene_metadata: "pd.DataFrame"
    track_metadata: "pd.DataFrame"
    space: str = "linear"

    @property
    def value_column(self) -> str:
        return "log_expression" if self.space == "log" else "count"

    def to_dataframe(self, long: bool = True) -> "pd.DataFrame":
        """Tidy long table: one row per (interval, gene, track) with metadata."""
        import pandas as pd

        b, g, c = self.counts.shape
        vals = self.counts.detach().float().cpu().numpy()
        gene_meta = self.gene_metadata.reset_index(drop=True)
        track_meta = self.track_metadata.reset_index(drop=True)

        rows = []
        for bi in range(b):
            for gi in range(g):
                grow = gene_meta.iloc[gi].to_dict()
                for ci in range(c):
                    trow = track_meta.iloc[ci].to_dict()
                    rec = {"batch": bi}
                    rec.update({f"gene_{k}": v for k, v in grow.items()})
                    rec.update({f"track_{k}": v for k, v in trow.items()})
                    rec[self.value_column] = vals[bi, gi, ci]
                    rows.append(rec)
        df = pd.DataFrame(rows)
        if not long:
            raise ValueError("Only long=True is supported; use to_tables() for a matrix.")
        return df

    def to_tables(self):
        """Return ``(X, obs, var)`` with ``X`` of shape ``[tracks, genes]``.

        Requires a single interval (``B == 1``); raises otherwise so the AnnData
        layout is unambiguous.
        """
        b = self.counts.shape[0]
        if b != 1:
            raise ValueError(
                f"to_tables()/to_anndata() require a single interval (B==1), got B={b}. "
                "Index a single window, e.g. gc.counts[i], or loop over intervals."
            )
        x = self.counts[0].detach().float().cpu().numpy().T  # [tracks, genes]
        obs = self.track_metadata.reset_index(drop=True).copy()
        var = self.gene_metadata.reset_index(drop=True).copy()
        return x, obs, var

    def to_anndata(self):
        """Build an ``anndata.AnnData`` (obs=tracks, var=genes, X=[tracks, genes]).

        ``obs_names`` are track names (falling back to track index) and
        ``var_names`` are gene ids — the scanpy convention. Setting them
        explicitly also avoids anndata's ``ImplicitModificationWarning`` about
        coercing an integer index to strings.
        """
        try:
            import anndata
        except ImportError:
            raise ImportError(
                "anndata is required for AnnData output. Install with: "
                "pip install anndata  (or pip install 'alphagenome-pytorch[inference-anndata]')"
            )
        x, obs, var = self.to_tables()
        obs = obs.copy()
        var = var.copy()
        obs.index = _string_index(obs, "track_name", "track_index")
        var.index = _string_index(var, "gene_id", None)
        return anndata.AnnData(X=x, obs=obs, var=var)


# --------------------------------------------------------------------------- #
# Shared metadata helpers
# --------------------------------------------------------------------------- #
def _track_metadata_frame(
    track_metadata: "Sequence[TrackMetadata] | None",
    num_tracks: int,
) -> "pd.DataFrame":
    import pandas as pd

    if track_metadata is None:
        return pd.DataFrame({"track_index": list(range(num_tracks))})
    if len(track_metadata) != num_tracks:
        raise ValueError(
            f"track_metadata has {len(track_metadata)} entries but predictions have "
            f"{num_tracks} tracks."
        )
    return pd.DataFrame([t.to_dict() for t in track_metadata])


def _validate_track_strands(strands: "Sequence[str]") -> None:
    """Reject strand labels outside ``{'+', '-', '.'}``.

    Unknown labels (``'plus'``, ``'?'``, …) would otherwise be silently treated as
    strand-incompatible by the matching logic, producing all-NaN gene rows.
    """
    invalid = sorted({str(s) for s in strands if str(s) not in ("+", "-", ".")})
    if invalid:
        raise ValueError(f"track strands must be '+', '-', or '.'; got invalid {invalid}")


def _track_strands(track_frame: "pd.DataFrame") -> list[str]:
    if "strand" not in track_frame.columns:
        raise ValueError(
            "strand-aware handling requires a 'strand' field on track_metadata "
            "(none found). Pass track_metadata with per-track strand, or use "
            "strand=None."
        )
    strands = [str(s) for s in track_frame["strand"].tolist()]
    _validate_track_strands(strands)
    return strands


def _apply_strand(
    counts: Tensor,                 # [B, G, C]
    gene_strands: Sequence[str],
    track_frame: "pd.DataFrame",
    strand: str | None,
):
    """Apply strand post-processing; returns (counts, track_frame)."""
    if strand in (None, "ignore", "all"):
        return counts, track_frame

    if strand == "match":
        track_strands = _track_strands(track_frame)
        gs = list(gene_strands)
        compat = torch.ones(len(gs), len(track_strands), dtype=torch.bool)
        for gi, g in enumerate(gs):
            for ci, t in enumerate(track_strands):
                compat[gi, ci] = (t == ".") or (g == ".") or (t == g)
        counts = counts.clone()
        counts[:, ~compat] = float("nan")
        return counts, track_frame

    if strand == "merge":
        return _merge_strand_pairs(counts, track_frame)

    raise ValueError(
        f"strand must be None/'ignore'/'all'/'match'/'merge', got {strand!r}"
    )


def _merge_strand_pairs(counts: Tensor, track_frame: "pd.DataFrame"):
    """Sum +/- track pairs sharing all metadata except strand → ~C/2 columns."""
    import pandas as pd

    if "strand" not in track_frame.columns:
        raise ValueError("strand='merge' requires a 'strand' field on track_metadata.")
    group_cols = [c for c in track_frame.columns if c not in ("strand", "track_index", "track_name")]
    if not group_cols:
        raise ValueError(
            "strand='merge' needs metadata fields besides strand to pair tracks "
            "(e.g. biosample/ontology/assay)."
        )
    b, g, _ = counts.shape
    groups: dict[tuple, list[int]] = {}
    order: list[tuple] = []
    for ci, row in track_frame.iterrows():
        key = tuple(row[col] for col in group_cols)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(ci)

    merged = counts.new_zeros(b, g, len(order))  # preserve device + dtype
    new_rows = []
    for out_i, key in enumerate(order):
        idxs = groups[key]
        merged[:, :, out_i] = counts[:, :, idxs].sum(dim=-1)
        base = track_frame.iloc[idxs[0]].to_dict()
        base["strand"] = "."
        base.pop("track_index", None)
        new_rows.append(base)
    new_frame = pd.DataFrame(new_rows)
    new_frame.insert(0, "track_index", list(range(len(order))))
    return merged, new_frame


# --------------------------------------------------------------------------- #
# aggregate_genes — gene-body counts
# --------------------------------------------------------------------------- #
def aggregate_genes(
    predictions: Tensor,
    gene_table: "pd.DataFrame",
    interval: tuple[str, int, int],
    *,
    track_metadata: "Sequence[TrackMetadata] | None" = None,
    reduce: str = "mean",
    strand: str | None = None,
) -> GeneCounts:
    """Aggregate predictions over gene **bodies** (exons + introns).

    Loss-style per-gene counts, using the same gene-body masks as the training
    gene-LFC loss (via :class:`GeneMaskExtractor`). Linear space.

    Args:
        predictions: ``[B, S, C]`` (or ``[S, C]``) RNA-seq tensor. ``S`` must span
            the interval at the tensor's resolution.
        gene_table: gene-body table from
            ``extensions.finetuning.gene_annotation.cached_load_gene_table``.
        interval: ``(chrom, start, end)`` the window covers (0-based half-open).
        track_metadata: per-track metadata for labels + strand.
        reduce: ``"mean"`` (length-normalized, default) or ``"sum"``.
        strand: post-processing of the ``[genes × tracks]`` matrix —
            ``None``/``"ignore"``/``"all"`` (no strand logic), ``"match"``
            (NaN incompatible cells), or ``"merge"`` (sum +/- track pairs).
    """
    from .extensions.finetuning.gene_annotation import GeneMaskExtractor

    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    chrom, start, end = interval
    seq_len = predictions.shape[1]
    resolution = (end - start) // seq_len
    if resolution < 1 or resolution * seq_len != (end - start):
        raise ValueError(
            f"interval width {end - start} is not divisible by sequence length {seq_len}."
        )

    extractor = GeneMaskExtractor(gene_table)
    body_mask_np, gene_meta = extractor.extract(chrom, start, end)  # [W, 2, G] at 1bp
    gene_strands = [str(s) for s in gene_meta["Strand"].tolist()] if len(gene_meta) else []

    mask = _bp_mask_to_bins(body_mask_np.any(axis=1), resolution, seq_len)  # [S, G]
    mask_t = torch.from_numpy(mask).to(predictions.device)

    # bin_size=resolution: 128bp predictions are bin sums, so "mean" must divide
    # by bases to stay per-base (see the module's Units note).
    counts = aggregate_intervals(
        predictions, mask_t, reduce=reduce, bin_size=resolution
    )  # [B, G, C]

    track_frame = _track_metadata_frame(track_metadata, predictions.shape[-1])
    counts, track_frame = _apply_strand(counts, gene_strands, track_frame, strand)
    gene_frame = _gene_metadata_frame(gene_meta)
    return GeneCounts(counts=counts, gene_metadata=gene_frame, track_metadata=track_frame, space="linear")


# --------------------------------------------------------------------------- #
# gene_expression — exon-based, log-space
# --------------------------------------------------------------------------- #
def gene_expression(
    predictions: Tensor,
    annotation: "GeneAnnotation",
    interval: tuple[str, int, int],
    *,
    track_metadata: "Sequence[TrackMetadata] | None" = None,
    log: str | None = "log1p",
    strand: str | None = "match",
    min_exon_fraction: float = 0.5,
    reduce: str = "mean",
) -> GeneCounts:
    """AlphaGenome gene expression: log mean coverage over annotated exons.

    Args:
        predictions: ``[B, S, C]`` (or ``[S, C]``) RNA-seq tensor in experimental
            (linear) space.
        annotation: a :class:`GeneAnnotation` built from a GTF/parquet that
            includes exon rows.
        interval: ``(chrom, start, end)`` (0-based half-open).
        track_metadata: per-track metadata (for labels + strand matching).
        log: ``"log1p"`` (default), ``"log"``, or ``None`` for linear.
        strand: default ``"match"`` (sense-strand expression). Same modes as
            :func:`aggregate_genes`.
        min_exon_fraction: keep a gene only if at least this fraction of its
            annotated exons fall fully within the interval — a count of whole
            exons, not a base-pair fraction.
        reduce: ``"mean"`` (default) or ``"sum"``.
    """
    per_gene = _exon_expression_matrix(
        predictions, annotation, interval,
        min_exon_fraction=min_exon_fraction, reduce=reduce, log=log,
    )
    counts = per_gene.counts           # [B, G, C]
    gene_strands = per_gene.gene_strands
    track_frame = _track_metadata_frame(track_metadata, counts.shape[-1])
    counts, track_frame = _apply_strand(counts, gene_strands, track_frame, strand)
    space = "log" if log else "linear"
    return GeneCounts(
        counts=counts,
        gene_metadata=per_gene.gene_frame,
        track_metadata=track_frame,
        space=space,
    )


_GENE_FRAME_COLUMNS = ["gene_id", "gene_name", "gene_type", "strand", "Start", "End"]


@dataclass
class _ExonWindow:
    """Cached per-window exon selection — the expensive annotation lookup.

    Holds only compact data: the kept genes' ids/strands/metadata and their exon
    ``[bin_start, bin_end)`` ranges. The dense ``[S, G]`` mask is rebuilt on
    demand (cheap tensor writes), so a cache of these stays small and device- /
    dtype-agnostic and can be reused across the pred/obs pair and across epochs.
    """

    gene_ids: list[str]
    gene_strands: list[str]
    gene_frame: "pd.DataFrame"
    bin_ranges: list[list[tuple[int, int]]]
    seq_len: int
    resolution: int

    def build_mask(self, device=None, dtype: torch.dtype = torch.float32) -> Tensor:
        """Reconstruct the ``[S, G]`` exon mask from the cached bin-ranges."""
        mask = torch.zeros(self.seq_len, len(self.gene_ids), dtype=dtype, device=device)
        for g, ranges in enumerate(self.bin_ranges):
            for b0, b1 in ranges:
                mask[b0:b1, g] = 1
        return mask


@dataclass
class _ExonExpr:
    counts: Tensor
    gene_frame: "pd.DataFrame"
    gene_strands: list[str]
    gene_ids: list[str]


def _window_resolution(interval: tuple[str, int, int], seq_len: int) -> int:
    _, start, end = interval
    resolution = (end - start) // seq_len
    if resolution < 1 or resolution * seq_len != (end - start):
        raise ValueError(
            f"interval width {end - start} is not divisible by sequence length {seq_len}."
        )
    return resolution


def _build_exon_window(
    annotation: "GeneAnnotation",
    interval: tuple[str, int, int],
    seq_len: int,
    *,
    min_exon_fraction: float,
) -> _ExonWindow:
    """Select genes in ``interval`` (≥``min_exon_fraction`` exon rule) + exon ranges.

    This is the only pandas-heavy step in the metric; :func:`_get_exon_window`
    memoizes it so it runs once per unique window across the whole run.
    """
    import pandas as pd
    from .variant_scoring.types import Interval

    resolution = _window_resolution(interval, seq_len)
    chrom, start, end = interval
    iv = Interval(chrom, start, end)

    kept_ids: list[str] = []
    kept_strands: list[str] = []
    kept_rows: list[dict] = []
    kept_ranges: list[list[tuple[int, int]]] = []

    for gid in annotation.get_genes_in_interval(iv):
        base = gid.split(".")[0]
        exons = annotation._get_exons_for_gene(base)
        if not exons:
            continue
        # Keep a gene when at least ``min_exon_fraction`` of its annotated exons
        # fall fully within the interval — a count of whole contained exons, not a
        # base-pair fraction. "Exons" here are the gene's *merged* exonic blocks
        # (overlapping/adjacent records collapsed across transcripts by
        # ``_get_exons_for_gene``), not raw GTF exon rows.
        n_within = sum(1 for s, e in exons if s >= start and e <= end)
        if (n_within / len(exons)) < min_exon_fraction:
            continue
        ranges = annotation.get_exon_bin_ranges(gid, iv, resolution, seq_len)
        if not ranges:
            continue
        info = annotation.get_gene_info(gid) or {}
        kept_ids.append(base)
        kept_strands.append(str(info.get("strand", ".")))
        kept_rows.append({
            "gene_id": info.get("gene_id", base),
            "gene_name": info.get("gene_name"),
            "gene_type": info.get("gene_type"),
            "strand": info.get("strand", "."),
            "Start": info.get("start"),
            "End": info.get("end"),
        })
        kept_ranges.append(ranges)

    gene_frame = (
        pd.DataFrame(kept_rows) if kept_rows
        else pd.DataFrame(columns=_GENE_FRAME_COLUMNS)
    )
    return _ExonWindow(
        gene_ids=kept_ids,
        gene_strands=kept_strands,
        gene_frame=gene_frame,
        bin_ranges=kept_ranges,
        seq_len=seq_len,
        resolution=resolution,
    )


def _get_exon_window(
    annotation: "GeneAnnotation",
    interval: tuple[str, int, int],
    seq_len: int,
    *,
    min_exon_fraction: float,
    cache: dict | None = None,
) -> _ExonWindow:
    """:func:`_build_exon_window` with optional memoization by window key.

    ``cache`` is a plain dict owned by the caller (e.g. the fine-tuning val loop
    creates one and reuses it every epoch). Keying on
    ``(chrom, start, end, seq_len, min_exon_fraction)`` makes the pandas lookup
    run once per unique window across the whole run, and lets the pred/obs pair
    share it within a window.
    """
    if cache is None:
        return _build_exon_window(annotation, interval, seq_len, min_exon_fraction=min_exon_fraction)
    key = (str(interval[0]), int(interval[1]), int(interval[2]), int(seq_len), float(min_exon_fraction))
    window = cache.get(key)
    if window is None:
        window = _build_exon_window(annotation, interval, seq_len, min_exon_fraction=min_exon_fraction)
        cache[key] = window
    return window


def _apply_log(counts: Tensor, log: str | None) -> Tensor:
    if log == "log1p":
        return torch.log1p(counts)
    if log == "log":
        return torch.log(counts + 1e-6)
    if log is None:
        return counts
    raise ValueError(f"log must be 'log1p', 'log', or None, got {log!r}")


def _aggregate_exon_window(
    predictions: Tensor,
    window: _ExonWindow,
    *,
    reduce: str,
    log: str | None,
) -> Tensor:
    """Aggregate ``predictions`` over a prebuilt exon window → ``[B, G, C]``."""
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if window.gene_ids:
        mask = window.build_mask(predictions.device, predictions.dtype)  # [S, G]
        counts = aggregate_intervals(
            predictions, mask, reduce=reduce, bin_size=window.resolution
        )  # [B, G, C]
    else:
        counts = predictions.new_zeros((predictions.shape[0], 0, predictions.shape[-1]))
    return _apply_log(counts, log)


def _exon_expression_matrix(
    predictions: Tensor,
    annotation: "GeneAnnotation",
    interval: tuple[str, int, int],
    *,
    min_exon_fraction: float,
    reduce: str,
    log: str | None,
    window: _ExonWindow | None = None,
) -> _ExonExpr:
    """Per-gene exon-mean (optionally log) matrix + gene metadata.

    Shared by :func:`gene_expression` (serving) and the fine-tuning validation
    metric. Applies the ≥``min_exon_fraction`` exon-containment rule; strand
    matching is left to the caller so both consumers stay in control. Pass a
    prebuilt ``window`` to skip the annotation lookup.
    """
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if window is None:
        window = _build_exon_window(
            annotation, interval, predictions.shape[1], min_exon_fraction=min_exon_fraction
        )
    counts = _aggregate_exon_window(predictions, window, reduce=reduce, log=log)
    return _ExonExpr(
        counts=counts,
        gene_frame=window.gene_frame,
        gene_strands=window.gene_strands,
        gene_ids=window.gene_ids,
    )


# --------------------------------------------------------------------------- #
# Validation-metric helpers (single-window values + cross-window combine)
# --------------------------------------------------------------------------- #
def gene_expression_values(
    predictions: Tensor,
    annotation: "GeneAnnotation",
    interval: tuple[str, int, int],
    *,
    min_exon_fraction: float = 0.5,
    log: str | None = "log1p",
    track_strands: "Sequence[str] | None" = None,
    window_cache: dict | None = None,
):
    """Per-gene exon expression for a **single** window (for the val metric).

    Returns ``(values, gene_ids, gene_strands)`` where ``values`` is ``[G, C]``
    (log-space by default). If ``track_strands`` is given, strand-incompatible
    ``(gene, track)`` cells are set to NaN (sense-strand matching).

    Bin size is derived from the interval width and the prediction sequence
    length, so predictions at any resolution work without being told which.

    Args:
        predictions: ``[S, C]`` or ``[1, S, C]`` predictions for one window.
        annotation: a :class:`GeneAnnotation` with exon rows.
        interval: ``(chrom, start, end)`` of this window.
        track_strands: per-track strand chars for strand matching (optional).
        window_cache: optional dict memoizing the per-window annotation lookup
            (the only pandas-heavy step). Pass the same dict for the pred and obs
            calls of a window, and reuse it across epochs, to build each window's
            exon selection exactly once. See :func:`_get_exon_window`.
    """
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if predictions.shape[0] != 1:
        raise ValueError("gene_expression_values expects a single window (B==1).")
    window = _get_exon_window(
        annotation, interval, predictions.shape[1],
        min_exon_fraction=min_exon_fraction, cache=window_cache,
    )
    counts = _aggregate_exon_window(predictions, window, reduce="mean", log=log)
    values = counts[0]  # [G, C]
    if track_strands is not None and values.numel() > 0:
        values = _strand_match_values(values, window.gene_strands, track_strands)
    return values, window.gene_ids, window.gene_strands


def _strand_match_values(
    values: Tensor,                 # [G, C]
    gene_strands: Sequence[str],
    track_strands: Sequence[str],
) -> Tensor:
    if values.shape[1] != len(track_strands):
        raise ValueError(
            f"track_strands has {len(track_strands)} entries but values have "
            f"{values.shape[1]} tracks."
        )
    gs = list(gene_strands)
    ts = [str(t) for t in track_strands]
    _validate_track_strands(ts)
    compat = torch.ones(len(gs), len(ts), dtype=torch.bool)
    for gi, g in enumerate(gs):
        for ci, t in enumerate(ts):
            compat[gi, ci] = (t == ".") or (g == ".") or (t == g)
    out = values.clone()
    out[~compat] = float("nan")
    return out


def combine_gene_expression(
    windows: "Sequence[tuple[Sequence[str], Tensor, Tensor]]",
    *,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Combine per-window ``(gene_ids, pred[G,C], obs[G,C])`` into gene-expression correlations.

    Genes are deduplicated by id across windows (first occurrence wins), avoiding
    duplicate genes across overlapping windows. Returns the three correlation
    flavors plus ``n_genes``.
    """
    seen: dict[str, int] = {}
    pred_rows: list[Tensor] = []
    obs_rows: list[Tensor] = []
    for gene_ids, pred, obs in windows:
        if pred.numel() == 0:
            continue
        for gi, gid in enumerate(gene_ids):
            if gid in seen:
                continue
            seen[gid] = len(pred_rows)
            pred_rows.append(pred[gi].float().cpu())
            obs_rows.append(obs[gi].float().cpu())
    if len(pred_rows) < 2:
        return {
            "across_genes": float("nan"),
            "across_genes_norm": float("nan"),
            "across_tracks_norm": float("nan"),
            "n_genes": len(pred_rows),
        }
    pred_m = torch.stack(pred_rows, dim=0)  # [G_total, C]
    obs_m = torch.stack(obs_rows, dim=0)
    result = gene_expression_correlations(pred_m, obs_m, eps=eps)
    result["n_genes"] = len(pred_rows)
    return result


# --------------------------------------------------------------------------- #
# Whole-chromosome streaming accumulator (tiled inference -> gene counts)
# --------------------------------------------------------------------------- #
class GeneCountAccumulator:
    """Accumulate per-gene counts from tiled whole-chromosome predictions.

    Feed each tile's predictions with their genomic coordinates via
    :meth:`add_tile`; the signal over each gene's exons (or gene body) is summed
    (or averaged) into a running ``[gene, track]`` matrix. A gene whose exons span
    several tiles is summed across all of them. Call :meth:`to_gene_counts` for a
    :class:`GeneCounts` (``.to_anndata()`` for an AnnData).

    Args:
        annotation: a :class:`GeneAnnotation`; exon rows required for
            ``over="exons"``.
        resolution: bp per prediction bin (1 or 128).
        over: ``"exons"`` (default) or ``"gene_body"``.
        reduce: ``"sum"`` (default, count-like) or ``"mean"`` (length-normalized).
    """

    def __init__(
        self,
        annotation: "GeneAnnotation",
        *,
        resolution: int,
        over: str = "exons",
        reduce: str = "sum",
    ) -> None:
        if over not in ("exons", "gene_body"):
            raise ValueError(f"over must be 'exons' or 'gene_body', got {over!r}")
        if reduce not in ("sum", "mean"):
            raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}")
        self.annotation = annotation
        self.resolution = int(resolution)
        self.over = over
        self.reduce = reduce
        self.n_tracks: int | None = None
        self._sum: dict[str, Any] = {}     # base gene id -> np.ndarray [n_tracks] running sum
        self._len: dict[str, float] = {}   # base gene id -> total masked bins (for mean)
        self._meta: dict[str, dict] = {}   # base gene id -> gene metadata row
        self._order: list[str] = []        # first-seen gene order

    @property
    def n_genes(self) -> int:
        return len(self._order)

    def add_tile(self, preds, chrom: str, start: int, end: int) -> None:
        """Accumulate one tile's kept-region predictions.

        Args:
            preds: ``[n_bins, n_tracks]`` predictions covering genomic
                ``[start, end)`` at ``self.resolution`` (``n_bins == (end-start)
                // resolution``). Tensor or ndarray.
            chrom, start, end: genomic span of ``preds`` (0-based half-open,
                resolution-aligned).
        """
        import numpy as np
        from .variant_scoring.types import Interval

        arr = preds.detach().cpu().numpy() if isinstance(preds, Tensor) else np.asarray(preds)
        if arr.ndim != 2:
            raise ValueError(f"preds must be [n_bins, n_tracks], got {arr.shape}")
        n_bins, c = arr.shape
        if self.n_tracks is None:
            self.n_tracks = c
        elif c != self.n_tracks:
            raise ValueError(f"tile has {c} tracks but accumulator holds {self.n_tracks}")

        iv = Interval(chrom, start, end)
        for gid in self.annotation.get_genes_in_interval(iv):
            if self.over == "exons":
                ranges = self.annotation.get_exon_bin_ranges(gid, iv, self.resolution, n_bins)
            else:
                ranges = self._body_bin_ranges(gid, start, end, n_bins)
            if not ranges:
                continue
            # Merge overlapping/adjacent bin ranges so a bin shared by two exons
            # (e.g. exons split by a short intron under coarse resolution) is
            # counted once, not once per exon.
            signal = np.zeros(c, dtype=np.float64)
            length = 0
            for b0, b1 in self.annotation._merge_intervals(ranges):
                signal += arr[b0:b1].sum(axis=0)
                length += b1 - b0
            if length == 0:
                continue
            base = gid.split(".")[0]
            if base not in self._sum:
                self._order.append(base)
                self._sum[base] = np.zeros(c, dtype=np.float64)
                self._len[base] = 0.0
                self._meta[base] = self._gene_meta_row(gid, base)
            self._sum[base] += signal
            self._len[base] += length

    def _body_bin_ranges(self, gid: str, start: int, end: int, n_bins: int):
        """Single ``[bin_start, bin_end)`` range for a gene body within the tile."""
        info = self.annotation.get_gene_info(gid)
        if not info or info.get("start") is None or info.get("end") is None:
            return []
        res, width = self.resolution, end - start
        rel_start = max(0, int(info["start"]) - start)
        rel_end = min(width, int(info["end"]) - start)
        if rel_start >= width or rel_end <= 0:
            return []
        b0 = max(0, min(rel_start // res, n_bins))
        b1 = max(0, min((rel_end + res - 1) // res, n_bins))
        return [(b0, b1)] if b0 < b1 else []

    def _gene_meta_row(self, gid: str, base: str) -> dict:
        info = self.annotation.get_gene_info(gid) or {}
        return {
            "gene_id": info.get("gene_id", base),
            "gene_name": info.get("gene_name"),
            "gene_type": info.get("gene_type"),
            "strand": info.get("strand", "."),
            "Start": info.get("start"),
            "End": info.get("end"),
        }

    def to_gene_counts(
        self,
        *,
        track_metadata: "Sequence[TrackMetadata] | pd.DataFrame | None" = None,
        log: bool = False,
        strand: str | None = None,
    ) -> GeneCounts:
        """Finalize accumulated signal into a single-interval :class:`GeneCounts`.

        Args:
            track_metadata: per-track metadata (``TrackMetadata`` sequence or a
                ready ``[C]``-row DataFrame) for the ``obs`` table and strand logic.
            log: if True, apply ``log1p`` after the reduce.
            strand: ``None``/``"match"``/``"merge"`` post-processing on the
                ``[gene × track]`` matrix (see :func:`aggregate_genes`).
        """
        import pandas as pd

        ids = self._order
        c = self.n_tracks or 0
        counts = torch.zeros(1, len(ids), c, dtype=torch.float32)
        for gi, base in enumerate(ids):
            s = torch.from_numpy(self._sum[base]).float()
            # `_len` counts bins; 128bp values are bin sums, so a per-base mean
            # divides by bases (see the module's Units note).
            n_bases = max(self._len[base] * self.resolution, 1.0)
            counts[0, gi] = s / n_bases if self.reduce == "mean" else s
        if log:
            counts = torch.log1p(counts)

        if isinstance(track_metadata, pd.DataFrame):
            # Same length check the TrackMetadata path gets via
            # _track_metadata_frame: a frame that doesn't describe the accumulated
            # tracks yields a GeneCounts whose counts and obs disagree, which only
            # surfaces later as a dimension error from inside anndata.
            if len(track_metadata) != c:
                raise ValueError(
                    f"track_metadata has {len(track_metadata)} rows but predictions "
                    f"have {c} tracks."
                )
            track_frame = track_metadata.reset_index(drop=True)
        else:
            track_frame = _track_metadata_frame(track_metadata, c)
        gene_frame = (
            pd.DataFrame([self._meta[b] for b in ids]) if ids
            else pd.DataFrame(columns=_GENE_FRAME_COLUMNS)
        )
        gene_strands = [self._meta[b]["strand"] for b in ids]
        counts, track_frame = _apply_strand(counts, gene_strands, track_frame, strand)
        return GeneCounts(
            counts=counts,
            gene_metadata=gene_frame,
            track_metadata=track_frame,
            space="log" if log else "linear",
        )


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def _string_index(frame: "pd.DataFrame", primary: str, fallback: str | None):
    """A unique string index for AnnData obs/var names (primary, else fallback)."""
    if primary in frame.columns:
        idx = frame[primary].astype(str)
        if idx.is_unique:
            return idx.values
    if fallback is not None and fallback in frame.columns:
        return frame[fallback].astype(str).values
    return frame.index.astype(str)


def _bp_mask_to_bins(bp_mask, resolution: int, seq_len: int):
    """Downsample a base-pair ``[W, G]`` bool mask to ``[S, G]`` bins (max-pool)."""
    import numpy as np

    if resolution == 1:
        return np.ascontiguousarray(bp_mask.astype(bool))
    w, g = bp_mask.shape
    usable = seq_len * resolution
    if w < usable:
        pad = np.zeros((usable - w, g), dtype=bool)
        bp_mask = np.concatenate([bp_mask, pad], axis=0)
    bp_mask = bp_mask[:usable]
    return bp_mask.reshape(seq_len, resolution, g).any(axis=1)


def _gene_metadata_frame(gene_meta: "pd.DataFrame") -> "pd.DataFrame":
    """Normalize a GeneMaskExtractor metadata slice to the GeneCounts var schema."""
    import pandas as pd

    if gene_meta is None or len(gene_meta) == 0:
        return pd.DataFrame(columns=["gene_id", "gene_name", "gene_type", "strand", "Start", "End"])
    out = pd.DataFrame({
        "gene_id": gene_meta.get("gene_id"),
        "gene_name": gene_meta.get("gene_name"),
        "gene_type": gene_meta.get("gene_type"),
        "strand": gene_meta.get("Strand"),
        "Start": gene_meta.get("Start"),
        "End": gene_meta.get("End"),
    })
    return out.reset_index(drop=True)

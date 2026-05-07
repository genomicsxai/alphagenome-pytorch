"""Variant scoring as a composable capability.

Holds the scoring-specific state (a :class:`VariantScoringModel` plus the
scorer-conversion logic) so the serving adapter doesn't need a subclass to
opt in. ``LocalDnaModelAdapter`` takes a :class:`VariantScorer` instance and
delegates ``score_*`` calls to it; if no scorer is configured, those calls
raise ``NotImplementedError``, which the REST layer maps to HTTP 501.
"""

from __future__ import annotations

import concurrent.futures
import importlib
import itertools
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from alphagenome.data import genome
from alphagenome.models import dna_output
from alphagenome.protos import dna_model_pb2

from alphagenome_pytorch.prediction import AlphaGenomePredictionRuntime
from alphagenome_pytorch.variant_scoring.inference import (
    VariantScoringModel,
    _build_ism_variants,
    get_recommended_scorers,
)
from alphagenome_pytorch.variant_scoring.scorers import (
    BaseVariantScorer as PTBaseVariantScorer,
    CenterMaskScorer as PTCenterMaskScorer,
    ContactMapScorer as PTContactMapScorer,
    GeneMaskActiveScorer as PTGeneMaskActiveScorer,
    GeneMaskLFCScorer as PTGeneMaskLFCScorer,
    GeneMaskSplicingScorer as PTGeneMaskSplicingScorer,
    PolyadenylationScorer as PTPolyadenylationScorer,
    SpliceJunctionScorer as PTSpliceJunctionScorer,
)
from alphagenome_pytorch.variant_scoring.types import (
    AggregationType as PTAggregationType,
    OutputType as PTOutputType,
    VariantScore,
)

from .adapter import (
    DEFAULT_MAX_WORKERS,
    MAX_VARIANT_SCORERS_PER_REQUEST,
    _OFFICIAL_TO_PT_OUTPUT,
    _as_numpy,
    _interval_to_pt,
    _normalize_output_type,
    _pt_metadata_to_track_df,
    _validate_sequence_length,
    _variant_to_pt,
)

ISM_NUCLEOTIDES = "ACGT"
_PT_AGGREGATION_BY_NAME = {a.name: a for a in PTAggregationType}


def _import_anndata_module():
    return importlib.import_module("anndata")


class VariantScorer:
    """Bundles a :class:`VariantScoringModel` with conversion helpers.

    Construct one and pass it as ``scorer=`` to :class:`LocalDnaModelAdapter`
    to enable variant-scoring routes. Without it, the adapter raises
    ``NotImplementedError`` from ``score_*`` calls.

    Holds a reference to the same ``runtime`` the adapter uses — both objects
    rely on it for sequence access and track metadata.
    """

    def __init__(
        self,
        runtime: AlphaGenomePredictionRuntime,
        scoring_model: VariantScoringModel,
    ):
        self.runtime = runtime
        self.scoring_model = scoring_model

    def score_variant(
        self,
        interval: genome.Interval,
        variant: genome.Variant,
        variant_scorers: Sequence[Any] = (),
        *,
        organism: Any = dna_model_pb2.ORGANISM_HOMO_SAPIENS,
    ) -> list[Any]:
        _validate_sequence_length(interval.width)
        organism_index = self.runtime.resolve_organism_index(organism)

        if not variant_scorers:
            organism_name = "human" if organism_index == 0 else "mouse"
            variant_scorers = list(get_recommended_scorers(organism_name))

        if len(variant_scorers) > MAX_VARIANT_SCORERS_PER_REQUEST:
            raise ValueError(
                f"Too many variant scorers requested: {len(variant_scorers)} "
                f"(max {MAX_VARIANT_SCORERS_PER_REQUEST})."
            )
        if len(variant_scorers) != len(set(map(str, variant_scorers))):
            raise ValueError(f"Duplicate variant scorers requested: {variant_scorers}.")

        local_scorers = [self._to_local_variant_scorer(vs) for vs in variant_scorers]
        scorer_results = self.scoring_model.score_variant(
            interval=_interval_to_pt(interval),
            variant=_variant_to_pt(variant),
            scorers=local_scorers,
            organism=organism_index,
        )

        return [
            self._scores_to_anndata(
                scores=local_result,
                organism_index=organism_index,
                fallback_variant_scorer=original_scorer,
                interval=interval,
                variant=variant,
            )
            for original_scorer, local_result in zip(
                variant_scorers, scorer_results, strict=True
            )
        ]

    def score_variants(
        self,
        intervals: genome.Interval | Sequence[genome.Interval],
        variants: Sequence[genome.Variant],
        variant_scorers: Sequence[Any] = (),
        *,
        organism: Any = dna_model_pb2.ORGANISM_HOMO_SAPIENS,
        progress_bar: bool = True,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> list[list[Any]]:
        if not isinstance(intervals, Sequence):
            intervals = [intervals] * len(variants)
        if len(intervals) != len(variants):
            raise ValueError(
                "Intervals and variants must have the same length. "
                f"Got {len(intervals)} intervals and {len(variants)} variants."
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.score_variant,
                    interval=interval,
                    variant=variant,
                    variant_scorers=variant_scorers,
                    organism=organism,
                )
                for interval, variant in zip(intervals, variants, strict=True)
            ]

            iterator: Iterable[Any]
            if progress_bar:
                try:
                    import tqdm.auto

                    iterator = tqdm.auto.tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc="Scoring variants",
                    )
                except ImportError:
                    iterator = concurrent.futures.as_completed(futures)
            else:
                iterator = concurrent.futures.as_completed(futures)

            for future in iterator:
                if (exc := future.exception()) is not None:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise exc

            return [future.result() for future in futures]

    def score_ism_variants(
        self,
        interval: genome.Interval,
        ism_interval: genome.Interval,
        variant_scorers: Sequence[Any] = (),
        *,
        organism: Any = dna_model_pb2.ORGANISM_HOMO_SAPIENS,
        interval_variant: genome.Variant | None = None,
        progress_bar: bool = True,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> list[list[Any]]:
        _validate_sequence_length(interval.width)
        if ism_interval.negative_strand:
            raise ValueError("ISM interval must be on the positive strand.")
        if ism_interval.chromosome != interval.chromosome:
            raise ValueError("ISM interval chromosome must match interval chromosome.")
        if ism_interval.start < interval.start or ism_interval.end > interval.end:
            raise ValueError("ISM interval must be contained within interval.")

        sequence = self.runtime.get_sequence(interval, variant=interval_variant)

        variants = _build_ism_variants(
            sequence=sequence,
            interval=interval,
            ism_interval=ism_interval,
            nucleotides=ISM_NUCLEOTIDES,
            variant_cls=genome.Variant,
        )

        if not variants:
            return []

        return self.score_variants(
            intervals=interval,
            variants=variants,
            variant_scorers=variant_scorers,
            organism=organism,
            progress_bar=progress_bar,
            max_workers=max_workers,
        )

    def _to_local_variant_scorer(self, scorer: Any) -> PTBaseVariantScorer:
        if isinstance(scorer, PTBaseVariantScorer):
            return scorer

        base = getattr(scorer, "base_variant_scorer", None)
        base_name = getattr(base, "name", None)
        class_name = scorer.__class__.__name__
        scorer_kind = (base_name or class_name or "").upper()
        requested_output = getattr(scorer, "requested_output", None)

        if scorer_kind in {"CENTER_MASK", "CENTERMASKSCORER"}:
            aggregation = getattr(scorer, "aggregation_type", None)
            aggregation_name = getattr(aggregation, "name", None)
            if aggregation_name is None:
                raise ValueError(f"Unsupported center-mask scorer: {scorer}")
            return PTCenterMaskScorer(
                requested_output=_OFFICIAL_TO_PT_OUTPUT[_normalize_output_type(requested_output)],
                width=getattr(scorer, "width", None),
                aggregation_type=_PT_AGGREGATION_BY_NAME[aggregation_name],
            )
        if scorer_kind in {"CONTACT_MAP", "CONTACTMAPSCORER"}:
            return PTContactMapScorer()
        if scorer_kind in {"GENE_MASK_LFC", "GENEMASKLFCSCORER"}:
            return PTGeneMaskLFCScorer(
                requested_output=_OFFICIAL_TO_PT_OUTPUT[_normalize_output_type(requested_output)],
                resolution=1,
            )
        if scorer_kind in {"GENE_MASK_ACTIVE", "GENEMASKACTIVESCORER"}:
            return PTGeneMaskActiveScorer(
                requested_output=_OFFICIAL_TO_PT_OUTPUT[_normalize_output_type(requested_output)],
                resolution=1,
            )
        if scorer_kind in {"GENE_MASK_SPLICING", "GENEMASKSPLICINGSCORER"}:
            return PTGeneMaskSplicingScorer(
                requested_output=_OFFICIAL_TO_PT_OUTPUT[_normalize_output_type(requested_output)],
                width=getattr(scorer, "width", None),
            )
        if scorer_kind in {"PA_QTL", "POLYADENYLATIONSCORER"}:
            return PTPolyadenylationScorer()
        if scorer_kind in {"SPLICE_JUNCTION", "SPLICEJUNCTIONSCORER"}:
            return PTSpliceJunctionScorer()

        which = getattr(scorer, "WhichOneof", None)
        if callable(which):
            field = scorer.WhichOneof("scorer")
            if field == "center_mask":
                center = scorer.center_mask
                return PTCenterMaskScorer(
                    requested_output=_OFFICIAL_TO_PT_OUTPUT[
                        dna_output.OutputType(center.requested_output)
                    ],
                    width=center.width if center.HasField("width") else None,
                    aggregation_type=_PT_AGGREGATION_BY_NAME[
                        dna_model_pb2.AggregationType.Name(center.aggregation_type)
                        .removeprefix("AGGREGATION_TYPE_")
                    ],
                )
            if field == "contact_map":
                return PTContactMapScorer()
            if field == "gene_mask":
                return PTGeneMaskLFCScorer(
                    requested_output=_OFFICIAL_TO_PT_OUTPUT[
                        dna_output.OutputType(scorer.gene_mask.requested_output)
                    ],
                    resolution=1,
                )
            if field == "gene_mask_active":
                return PTGeneMaskActiveScorer(
                    requested_output=_OFFICIAL_TO_PT_OUTPUT[
                        dna_output.OutputType(scorer.gene_mask_active.requested_output)
                    ],
                    resolution=1,
                )
            if field == "gene_mask_splicing":
                return PTGeneMaskSplicingScorer(
                    requested_output=_OFFICIAL_TO_PT_OUTPUT[
                        dna_output.OutputType(scorer.gene_mask_splicing.requested_output)
                    ],
                    width=scorer.gene_mask_splicing.width
                    if scorer.gene_mask_splicing.HasField("width")
                    else None,
                )
            if field == "pa_qtl":
                return PTPolyadenylationScorer()
            if field == "splice_junction":
                return PTSpliceJunctionScorer()

        raise ValueError(f"Unsupported variant scorer type: {scorer!r}")

    def _scores_to_anndata(
        self,
        *,
        scores: VariantScore | list[VariantScore],
        organism_index: int,
        fallback_variant_scorer: Any,
        interval: genome.Interval,
        variant: genome.Variant,
    ):
        anndata = _import_anndata_module()

        score_list = scores if isinstance(scores, list) else [scores]
        if not score_list:
            return anndata.AnnData(
                X=np.zeros((0, 0), dtype=np.float32),
                obs=pd.DataFrame(),
                var=pd.DataFrame(columns=["name", "strand"]),
                uns={
                    "interval": interval,
                    "variant": variant,
                    "variant_scorer": fallback_variant_scorer,
                },
            )

        x_rows = []
        obs_rows = []
        has_gene_metadata = False
        for score in score_list:
            values = _as_numpy(score.scores).astype(np.float32, copy=False)
            if values.ndim != 1:
                values = values.reshape(-1)
            x_rows.append(values)

            row = {
                "gene_id": score.gene_id,
                "gene_name": score.gene_name,
                "gene_type": score.gene_type,
                "strand": score.gene_strand,
                "junction_Start": score.junction_start,
                "junction_End": score.junction_end,
            }
            if any(v is not None for v in row.values()):
                has_gene_metadata = True
            obs_rows.append(row)

        X = np.stack(x_rows, axis=0).astype(np.float32, copy=False)
        n_tracks = X.shape[1] if X.ndim == 2 else 0

        scorer_for_output = score_list[0].scorer if score_list else None
        requested_output = getattr(scorer_for_output, "requested_output", None)
        if isinstance(requested_output, PTOutputType):
            pt_output = requested_output
        else:
            pt_output = _OFFICIAL_TO_PT_OUTPUT.get(
                _normalize_output_type(
                    getattr(
                        fallback_variant_scorer,
                        "requested_output",
                        dna_output.OutputType.RNA_SEQ,
                    )
                ),
                PTOutputType.RNA_SEQ,
            )
        track_metadata = self.runtime.get_track_metadata(
            organism_index,
            output_name=pt_output.value,
        )
        var_df = _pt_metadata_to_track_df(track_metadata, num_tracks=n_tracks)

        if has_gene_metadata:
            obs_df = pd.DataFrame(obs_rows)
            obs_df.index = obs_df.index.map(str)
        else:
            obs_df = pd.DataFrame(index=[str(i) for i in range(X.shape[0])])

        var_df = var_df.reset_index(drop=True)
        var_df.index = var_df.index.map(str)

        return anndata.AnnData(
            X=X,
            obs=obs_df,
            var=var_df,
            uns={
                "interval": interval,
                "variant": variant,
                "variant_scorer": fallback_variant_scorer,
            },
        )


__all__ = ["VariantScorer"]

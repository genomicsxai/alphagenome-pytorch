"""Local serving adapter for AlphaGenome notebook-compatible APIs.

This module provides a local implementation of the notebook-facing AlphaGenome
surface (`predict_*`, `score_*`, `output_metadata`) by wrapping
`VariantScoringModel` and converting outputs to upstream-compatible containers.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import importlib
import itertools
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

from alphagenome.data import genome
from alphagenome.data import junction_data as ag_junction_data
from alphagenome.data import track_data as ag_track_data
from alphagenome.models import dna_output
from alphagenome.protos import dna_model_pb2

from alphagenome_pytorch.variant_scoring.inference import VariantScoringModel, get_recommended_scorers
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
    Interval as PTInterval,
    OutputType as PTOutputType,
    TrackMetadata as PTTrackMetadata,
    Variant as PTVariant,
    VariantScore,
)
from alphagenome_pytorch.utils.splicing import unstack_junction_predictions

# Supported DNA sequence lengths, matching upstream dna_client constants.
SEQUENCE_LENGTH_16KB = 2**14  # 16_384
SEQUENCE_LENGTH_100KB = 2**17  # 131_072
SEQUENCE_LENGTH_500KB = 2**19  # 524_288
SEQUENCE_LENGTH_1MB = 2**20  # 1_048_576

SUPPORTED_SEQUENCE_LENGTHS: Mapping[str, int] = {
    'SEQUENCE_LENGTH_16KB': SEQUENCE_LENGTH_16KB,
    'SEQUENCE_LENGTH_100KB': SEQUENCE_LENGTH_100KB,
    'SEQUENCE_LENGTH_500KB': SEQUENCE_LENGTH_500KB,
    'SEQUENCE_LENGTH_1MB': SEQUENCE_LENGTH_1MB,
}

MAX_VARIANT_SCORERS_PER_REQUEST = 20
DEFAULT_MAX_WORKERS = 5
VALID_SEQUENCE_CHARACTERS = frozenset('ACGTN')
ISM_NUCLEOTIDES = 'ACGT'

_ORGANISM_TO_INDEX = {
    dna_model_pb2.ORGANISM_HOMO_SAPIENS: 0,
    dna_model_pb2.ORGANISM_MUS_MUSCULUS: 1,
    'HOMO_SAPIENS': 0,
    'MUS_MUSCULUS': 1,
    'human': 0,
    'mouse': 1,
    0: 0,
    1: 1,
}

_PT_OUTPUT_TO_OFFICIAL = {
    PTOutputType.ATAC: dna_output.OutputType.ATAC,
    PTOutputType.CAGE: dna_output.OutputType.CAGE,
    PTOutputType.DNASE: dna_output.OutputType.DNASE,
    PTOutputType.RNA_SEQ: dna_output.OutputType.RNA_SEQ,
    PTOutputType.CHIP_HISTONE: dna_output.OutputType.CHIP_HISTONE,
    PTOutputType.CHIP_TF: dna_output.OutputType.CHIP_TF,
    PTOutputType.SPLICE_SITES: dna_output.OutputType.SPLICE_SITES,
    PTOutputType.SPLICE_SITE_USAGE: dna_output.OutputType.SPLICE_SITE_USAGE,
    PTOutputType.SPLICE_JUNCTIONS: dna_output.OutputType.SPLICE_JUNCTIONS,
    PTOutputType.CONTACT_MAPS: dna_output.OutputType.CONTACT_MAPS,
    PTOutputType.PROCAP: dna_output.OutputType.PROCAP,
}
_OFFICIAL_TO_PT_OUTPUT = {v: k for k, v in _PT_OUTPUT_TO_OFFICIAL.items()}

_OFFICIAL_OUTPUT_FIELD = {
    dna_output.OutputType.ATAC: 'atac',
    dna_output.OutputType.CAGE: 'cage',
    dna_output.OutputType.DNASE: 'dnase',
    dna_output.OutputType.RNA_SEQ: 'rna_seq',
    dna_output.OutputType.CHIP_HISTONE: 'chip_histone',
    dna_output.OutputType.CHIP_TF: 'chip_tf',
    dna_output.OutputType.SPLICE_SITES: 'splice_sites',
    dna_output.OutputType.SPLICE_SITE_USAGE: 'splice_site_usage',
    dna_output.OutputType.SPLICE_JUNCTIONS: 'splice_junctions',
    dna_output.OutputType.CONTACT_MAPS: 'contact_maps',
    dna_output.OutputType.PROCAP: 'procap',
}

_PRODUCED_PT_OUTPUTS = {
    PTOutputType.ATAC,
    PTOutputType.CAGE,
    PTOutputType.DNASE,
    PTOutputType.RNA_SEQ,
    PTOutputType.CHIP_HISTONE,
    PTOutputType.CHIP_TF,
    PTOutputType.SPLICE_SITES,
    PTOutputType.SPLICE_SITE_USAGE,
    PTOutputType.SPLICE_JUNCTIONS,
    PTOutputType.CONTACT_MAPS,
    PTOutputType.PROCAP,
}

_PT_AGGREGATION_BY_NAME = {a.name: a for a in PTAggregationType}


def _validate_sequence_length(length: int) -> None:
    if length not in SUPPORTED_SEQUENCE_LENGTHS.values():
        raise ValueError(
            f'Sequence length {length} not supported. '
            f'Supported lengths: {list(SUPPORTED_SEQUENCE_LENGTHS.values())}'
        )


def _resolve_organism_index(organism: Any) -> int:
    if organism is None:
        return 0
    if hasattr(organism, 'value'):
        candidate = getattr(organism, 'value')
        if candidate in _ORGANISM_TO_INDEX:
            return _ORGANISM_TO_INDEX[candidate]
    if hasattr(organism, 'name'):
        candidate = getattr(organism, 'name')
        if candidate in _ORGANISM_TO_INDEX:
            return _ORGANISM_TO_INDEX[candidate]
        if isinstance(candidate, str) and candidate.startswith('ORGANISM_'):
            stripped = candidate.removeprefix('ORGANISM_')
            if stripped in _ORGANISM_TO_INDEX:
                return _ORGANISM_TO_INDEX[stripped]
    if organism in _ORGANISM_TO_INDEX:
        return _ORGANISM_TO_INDEX[organism]
    if isinstance(organism, str):
        normalized = organism.upper().removeprefix('ORGANISM_')
        if normalized in _ORGANISM_TO_INDEX:
            return _ORGANISM_TO_INDEX[normalized]
        lower = organism.lower()
        if lower in _ORGANISM_TO_INDEX:
            return _ORGANISM_TO_INDEX[lower]
    raise ValueError(
        f'Unsupported organism "{organism}". Expected Homo sapiens or Mus musculus.'
    )


def _organism_proto_from_index(idx: int) -> int:
    if idx == 0:
        return dna_model_pb2.ORGANISM_HOMO_SAPIENS
    if idx == 1:
        return dna_model_pb2.ORGANISM_MUS_MUSCULUS
    raise ValueError(f'Unsupported organism index: {idx}')


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _import_anndata_module():
    # Import lazily to keep serving optional and avoid importing at module load.
    return importlib.import_module('anndata')


def _interval_to_pt(interval: genome.Interval) -> PTInterval:
    return PTInterval(
        chromosome=interval.chromosome,
        start=interval.start,
        end=interval.end,
        strand=interval.strand,
        name=interval.name,
    )


def _variant_to_pt(variant: genome.Variant) -> PTVariant:
    return PTVariant(
        chromosome=variant.chromosome,
        position=variant.position,
        reference_bases=variant.reference_bases,
        alternate_bases=variant.alternate_bases,
        name=variant.name,
    )


def _normalize_output_type(value: Any) -> dna_output.OutputType:
    if isinstance(value, dna_output.OutputType):
        return value
    if isinstance(value, int):
        return dna_output.OutputType(value)
    if hasattr(value, 'name'):
        name = getattr(value, 'name')
        if isinstance(name, str):
            normalized = name.upper().removeprefix('OUTPUT_TYPE_')
            return dna_output.OutputType[normalized]
    if isinstance(value, str):
        normalized = value.upper().removeprefix('OUTPUT_TYPE_')
        if normalized in dna_output.OutputType.__members__:
            return dna_output.OutputType[normalized]
    raise ValueError(f'Unsupported output type value: {value}')


def _normalize_requested_outputs(
    requested_outputs: Iterable[Any],
) -> list[dna_output.OutputType]:
    outputs: list[dna_output.OutputType] = []
    seen: set[dna_output.OutputType] = set()
    for output_type in requested_outputs:
        normalized = _normalize_output_type(output_type)
        if normalized not in seen:
            outputs.append(normalized)
            seen.add(normalized)
    return outputs


def _normalize_ontology_terms(ontology_terms: Iterable[Any] | None) -> list[str] | None:
    if ontology_terms is None:
        return None
    normalized: list[str] = []
    for term in ontology_terms:
        if term is None:
            continue
        if isinstance(term, str):
            normalized.append(term)
            continue
        # OntologyTerm objects from alphagenome.data.ontology expose ontology_curie.
        curie = getattr(term, 'ontology_curie', None)
        if curie is not None:
            normalized.append(str(curie))
            continue
        tid = getattr(term, 'id', None)
        if tid is not None:
            normalized.append(str(tid))
            continue
        normalized.append(str(term))
    # Preserve order while dropping duplicates.
    return list(dict.fromkeys(normalized))


class LocalDnaModelAdapter:
    """Notebook-compatible local model adapter.

    This class mirrors the notebook-critical subset of the AlphaGenome API:
    `predict_sequence`, `predict_interval`, `predict_variant`, `score_variant`,
    `score_variants`, `score_ism_variants`, and `output_metadata`.
    """

    def __init__(self, scoring_model: VariantScoringModel):
        self.scoring_model = scoring_model

    def predict_sequence(
        self,
        sequence: str,
        *,
        organism: Any = dna_model_pb2.ORGANISM_HOMO_SAPIENS,
        requested_outputs: Iterable[Any],
        ontology_terms: Iterable[Any] | None = None,
        interval: genome.Interval | None = None,
    ) -> dna_output.Output:
        invalid = set(sequence) - VALID_SEQUENCE_CHARACTERS
        if invalid:
            bad = ','.join(sorted(invalid))
            raise ValueError(
                f'Invalid DNA sequence. Allowed characters are A/C/G/T/N. Found: {bad}'
            )
        _validate_sequence_length(len(sequence))
        organism_index = _resolve_organism_index(organism)
        outputs = self.scoring_model.predict(sequence, organism=organism_index)
        requested = _normalize_requested_outputs(requested_outputs)
        ontology = _normalize_ontology_terms(ontology_terms)
        return self._convert_output(
            outputs,
            organism_index=organism_index,
            requested_outputs=requested,
            interval=interval,
            ontology_terms=ontology,
        )

    def predict_interval(
        self,
        interval: genome.Interval,
        *,
        organism: Any = dna_model_pb2.ORGANISM_HOMO_SAPIENS,
        requested_outputs: Iterable[Any],
        ontology_terms: Iterable[Any] | None = None,
    ) -> dna_output.Output:
        _validate_sequence_length(interval.width)
        pt_interval = _interval_to_pt(interval)
        sequence = self.scoring_model.get_sequence(pt_interval)
        return self.predict_sequence(
            sequence=sequence,
            organism=organism,
            requested_outputs=requested_outputs,
            ontology_terms=ontology_terms,
            interval=interval,
        )

    def predict_variant(
        self,
        interval: genome.Interval,
        variant: genome.Variant,
        *,
        organism: Any = dna_model_pb2.ORGANISM_HOMO_SAPIENS,
        requested_outputs: Iterable[Any],
        ontology_terms: Iterable[Any] | None = None,
    ) -> dna_output.VariantOutput:
        _validate_sequence_length(interval.width)
        organism_index = _resolve_organism_index(organism)
        pt_interval = _interval_to_pt(interval)
        pt_variant = _variant_to_pt(variant)

        ref_outputs, alt_outputs = self.scoring_model.predict_variant(
            interval=pt_interval,
            variant=pt_variant,
            organism=organism_index,
        )

        requested = _normalize_requested_outputs(requested_outputs)
        ontology = _normalize_ontology_terms(ontology_terms)
        return dna_output.VariantOutput(
            reference=self._convert_output(
                ref_outputs,
                organism_index=organism_index,
                requested_outputs=requested,
                interval=interval,
                ontology_terms=ontology,
            ),
            alternate=self._convert_output(
                alt_outputs,
                organism_index=organism_index,
                requested_outputs=requested,
                interval=interval,
                ontology_terms=ontology,
            ),
        )

    def score_interval(self, *args, **kwargs):
        del args, kwargs  # Unused for the local serving MVP.
        raise NotImplementedError('score_interval is not implemented in LocalDnaModelAdapter.')

    def score_variant(
        self,
        interval: genome.Interval,
        variant: genome.Variant,
        variant_scorers: Sequence[Any] = (),
        *,
        organism: Any = dna_model_pb2.ORGANISM_HOMO_SAPIENS,
    ) -> list[Any]:
        _validate_sequence_length(interval.width)
        organism_index = _resolve_organism_index(organism)

        if not variant_scorers:
            organism_name = 'human' if organism_index == 0 else 'mouse'
            variant_scorers = list(get_recommended_scorers(organism_name))

        if len(variant_scorers) > MAX_VARIANT_SCORERS_PER_REQUEST:
            raise ValueError(
                f'Too many variant scorers requested: {len(variant_scorers)} '
                f'(max {MAX_VARIANT_SCORERS_PER_REQUEST}).'
            )
        if len(variant_scorers) != len(set(map(str, variant_scorers))):
            raise ValueError(f'Duplicate variant scorers requested: {variant_scorers}.')

        local_scorers = [self._to_local_variant_scorer(vs) for vs in variant_scorers]
        pt_interval = _interval_to_pt(interval)
        pt_variant = _variant_to_pt(variant)

        scorer_results = self.scoring_model.score_variant(
            interval=pt_interval,
            variant=pt_variant,
            scorers=local_scorers,
            organism=organism_index,
        )

        outputs = []
        for original_scorer, local_result in zip(
            variant_scorers, scorer_results, strict=True
        ):
            adata = self._scores_to_anndata(
                scores=local_result,
                organism_index=organism_index,
                fallback_variant_scorer=original_scorer,
                interval=interval,
                variant=variant,
            )
            outputs.append(adata)
        return outputs

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
                'Intervals and variants must have the same length. '
                f'Got {len(intervals)} intervals and {len(variants)} variants.'
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
                        desc='Scoring variants',
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
            raise ValueError('ISM interval must be on the positive strand.')
        if ism_interval.chromosome != interval.chromosome:
            raise ValueError('ISM interval chromosome must match interval chromosome.')
        if ism_interval.start < interval.start or ism_interval.end > interval.end:
            raise ValueError('ISM interval must be contained within interval.')

        pt_interval = _interval_to_pt(interval)
        pt_interval_variant = _variant_to_pt(interval_variant) if interval_variant else None
        sequence = self.scoring_model.get_sequence(pt_interval, variant=pt_interval_variant)

        variants: list[genome.Variant] = []
        for genomic_pos_0b in range(ism_interval.start, ism_interval.end):
            rel = genomic_pos_0b - interval.start
            if rel < 0 or rel >= len(sequence):
                continue
            ref_base = sequence[rel].upper()
            if ref_base not in ISM_NUCLEOTIDES:
                continue
            for alt_base in ISM_NUCLEOTIDES:
                if alt_base == ref_base:
                    continue
                variants.append(
                    genome.Variant(
                        chromosome=interval.chromosome,
                        position=genomic_pos_0b + 1,
                        reference_bases=ref_base,
                        alternate_bases=alt_base,
                    )
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

    def output_metadata(
        self,
        organism: Any = dna_model_pb2.ORGANISM_HOMO_SAPIENS,
    ) -> dna_output.OutputMetadata:
        organism_index = _resolve_organism_index(organism)
        track_metadata = self.scoring_model.get_track_metadata(organism_index)

        metadata_kwargs: dict[str, pd.DataFrame | None] = {
            field.name: None for field in dataclasses.fields(dna_output.OutputMetadata)
        }
        for official_output_type, field_name in _OFFICIAL_OUTPUT_FIELD.items():
            pt_output_type = _OFFICIAL_TO_PT_OUTPUT[official_output_type]
            entries = track_metadata.get(pt_output_type, [])
            if not entries:
                continue
            if official_output_type == dna_output.OutputType.SPLICE_JUNCTIONS:
                metadata_kwargs[field_name] = self._pt_metadata_to_junction_df(entries)
            else:
                metadata_kwargs[field_name] = self._pt_metadata_to_track_df(entries)
        return dna_output.OutputMetadata(**metadata_kwargs)

    def _extract_head_output(
        self,
        outputs: Mapping[str, Any],
        pt_output_type: PTOutputType,
    ) -> tuple[np.ndarray, int]:
        key = pt_output_type.value
        if key not in outputs:
            raise KeyError(f'Requested output "{key}" not found in model outputs.')
        raw = outputs[key]

        if pt_output_type == PTOutputType.SPLICE_SITES and isinstance(raw, Mapping):
            array = _as_numpy(raw.get('probs'))
            return self._squeeze_batch(array), 1
        if pt_output_type == PTOutputType.SPLICE_SITE_USAGE and isinstance(raw, Mapping):
            array = _as_numpy(raw.get('predictions'))
            return self._squeeze_batch(array), 1

        if isinstance(raw, Mapping):
            # Standard multi-resolution heads: prefer 1bp when available.
            if 1 in raw:
                return self._squeeze_batch(_as_numpy(raw[1])), 1
            if 128 in raw:
                return self._squeeze_batch(_as_numpy(raw[128])), 128
            first_key = next(iter(raw))
            return self._squeeze_batch(_as_numpy(raw[first_key])), int(first_key)

        resolution = 128 if pt_output_type in {
            PTOutputType.CONTACT_MAPS,
            PTOutputType.CHIP_TF,
            PTOutputType.CHIP_HISTONE,
        } else 1
        return self._squeeze_batch(_as_numpy(raw)), resolution

    def _convert_output(
        self,
        outputs: Mapping[str, Any],
        *,
        organism_index: int,
        requested_outputs: Sequence[dna_output.OutputType],
        interval: genome.Interval | None,
        ontology_terms: Sequence[str] | None,
    ) -> dna_output.Output:
        output_kwargs: dict[str, Any] = {field: None for field in _OFFICIAL_OUTPUT_FIELD.values()}
        for official_output_type in requested_outputs:
            pt_output_type = _OFFICIAL_TO_PT_OUTPUT.get(official_output_type)
            if pt_output_type is None or pt_output_type not in _PRODUCED_PT_OUTPUTS:
                continue

            field_name = _OFFICIAL_OUTPUT_FIELD[official_output_type]

            if official_output_type == dna_output.OutputType.SPLICE_JUNCTIONS:
                output_kwargs[field_name] = self._build_junction_data(
                    outputs=outputs,
                    organism_index=organism_index,
                    interval=interval,
                    ontology_terms=ontology_terms,
                )
                continue

            values, resolution = self._extract_head_output(outputs, pt_output_type)
            if values.ndim == 1:
                values = values[:, None]
            metadata = self._build_track_metadata_df(
                pt_output_type=pt_output_type,
                organism_index=organism_index,
                num_tracks=values.shape[-1],
            )

            if ontology_terms and 'ontology_curie' in metadata.columns:
                keep_mask = metadata['ontology_curie'].isin(ontology_terms).to_numpy()
                metadata = metadata.loc[keep_mask].reset_index(drop=True)
                values = values[..., keep_mask]

            output_kwargs[field_name] = ag_track_data.TrackData(
                values=values,
                metadata=metadata,
                resolution=resolution,
                interval=interval,
            )
        return dna_output.Output(**output_kwargs)

    def _build_junction_data(
        self,
        *,
        outputs: Mapping[str, Any],
        organism_index: int,
        interval: genome.Interval | None,
        ontology_terms: Sequence[str] | None,
    ) -> ag_junction_data.JunctionData | None:
        key = PTOutputType.SPLICE_JUNCTIONS.value
        if key not in outputs:
            return None

        raw = outputs[key]
        if not isinstance(raw, Mapping):
            return None
        if 'pred_counts' not in raw or 'splice_site_positions' not in raw:
            return None

        pred_counts = _as_numpy(raw['pred_counts'])
        positions = _as_numpy(raw['splice_site_positions'])
        pred_counts = self._ensure_batch(pred_counts)
        positions = self._ensure_batch(positions)

        interval_start = interval.start if interval is not None else 0
        scores, starts, ends, strands, valid_mask = unstack_junction_predictions(
            torch.as_tensor(pred_counts),
            torch.as_tensor(positions),
            interval_start=interval_start,
        )

        scores = _as_numpy(scores)[0]
        starts = _as_numpy(starts)[0]
        ends = _as_numpy(ends)[0]
        strands = _as_numpy(strands)[0]
        valid_mask = _as_numpy(valid_mask)[0].astype(bool)

        if scores.ndim != 2:
            return None

        metadata = self._build_junction_metadata_df(
            organism_index=organism_index,
            num_tracks=scores.shape[-1],
        )
        if ontology_terms and 'ontology_curie' in metadata.columns:
            keep_mask = metadata['ontology_curie'].isin(ontology_terms).to_numpy()
            metadata = metadata.loc[keep_mask].reset_index(drop=True)
            scores = scores[:, keep_mask]

        junction_intervals = []
        for start, end, strand, keep in zip(starts, ends, strands, valid_mask, strict=True):
            if not keep:
                continue
            if end <= start:
                continue
            strand_symbol = '+' if int(strand) == 0 else '-'
            chrom = interval.chromosome if interval is not None else 'chrNA'
            junction_intervals.append(
                genome.Interval(chromosome=chrom, start=int(start), end=int(end), strand=strand_symbol)
            )

        if junction_intervals:
            junctions = np.asarray(junction_intervals, dtype=object)
            values = scores[valid_mask]
        else:
            junctions = np.asarray([], dtype=object)
            values = np.zeros((0, len(metadata)), dtype=np.float32)

        return ag_junction_data.JunctionData(
            junctions=junctions,
            values=values,
            metadata=metadata,
            interval=interval,
        )

    def _build_track_metadata_df(
        self,
        *,
        pt_output_type: PTOutputType,
        organism_index: int,
        num_tracks: int,
    ) -> pd.DataFrame:
        metadata = self.scoring_model.get_track_metadata(organism_index).get(pt_output_type, [])
        return self._pt_metadata_to_track_df(metadata, num_tracks=num_tracks)

    def _build_junction_metadata_df(
        self,
        *,
        organism_index: int,
        num_tracks: int,
    ) -> pd.DataFrame:
        metadata = self.scoring_model.get_track_metadata(organism_index).get(
            PTOutputType.SPLICE_JUNCTIONS, []
        )
        return self._pt_metadata_to_junction_df(metadata, num_tracks=num_tracks)

    def _pt_metadata_to_track_df(
        self,
        metadata: Sequence[PTTrackMetadata],
        num_tracks: int | None = None,
    ) -> pd.DataFrame:
        rows = []
        for i, meta in enumerate(metadata):
            rows.append(
                {
                    'name': meta.track_name or f'track_{i}',
                    'strand': meta.track_strand or '.',
                    'ontology_curie': meta.ontology_curie,
                    'gtex_tissue': meta.gtex_tissue,
                    'Assay title': meta.assay_title,
                    'biosample_name': meta.biosample_name,
                    'biosample_type': meta.biosample_type,
                    'transcription_factor': meta.transcription_factor,
                    'histone_mark': meta.histone_mark,
                }
            )
        if not rows and num_tracks is not None:
            rows = [{'name': f'track_{i}', 'strand': '.'} for i in range(num_tracks)]
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=['name', 'strand'])
        if num_tracks is not None:
            if len(df) < num_tracks:
                for i in range(len(df), num_tracks):
                    df.loc[i] = {'name': f'track_{i}', 'strand': '.'}
            elif len(df) > num_tracks:
                df = df.iloc[:num_tracks].copy()
        df = df.reset_index(drop=True)
        if 'name' not in df.columns:
            df['name'] = [f'track_{i}' for i in range(len(df))]
        if 'strand' not in df.columns:
            df['strand'] = '.'
        return df

    def _pt_metadata_to_junction_df(
        self,
        metadata: Sequence[PTTrackMetadata],
        num_tracks: int | None = None,
    ) -> pd.DataFrame:
        rows = []
        for i, meta in enumerate(metadata):
            rows.append(
                {
                    'name': meta.track_name or f'track_{i}',
                    'ontology_curie': meta.ontology_curie,
                    'gtex_tissue': meta.gtex_tissue,
                    'Assay title': meta.assay_title,
                    'biosample_name': meta.biosample_name,
                    'biosample_type': meta.biosample_type,
                }
            )
        if not rows and num_tracks is not None:
            rows = [{'name': f'track_{i}'} for i in range(num_tracks)]
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=['name'])
        if num_tracks is not None:
            if len(df) < num_tracks:
                for i in range(len(df), num_tracks):
                    df.loc[i] = {'name': f'track_{i}'}
            elif len(df) > num_tracks:
                df = df.iloc[:num_tracks].copy()
        return df.reset_index(drop=True)

    def _to_local_variant_scorer(self, scorer: Any) -> PTBaseVariantScorer:
        if isinstance(scorer, PTBaseVariantScorer):
            return scorer

        base = getattr(scorer, 'base_variant_scorer', None)
        base_name = getattr(base, 'name', None)
        class_name = scorer.__class__.__name__
        scorer_kind = (base_name or class_name or '').upper()
        requested_output = getattr(scorer, 'requested_output', None)

        if scorer_kind in {'CENTER_MASK', 'CENTERMASKSCORER'}:
            aggregation = getattr(scorer, 'aggregation_type', None)
            aggregation_name = getattr(aggregation, 'name', None)
            if aggregation_name is None:
                raise ValueError(f'Unsupported center-mask scorer: {scorer}')
            return PTCenterMaskScorer(
                requested_output=_OFFICIAL_TO_PT_OUTPUT[_normalize_output_type(requested_output)],
                width=getattr(scorer, 'width', None),
                aggregation_type=_PT_AGGREGATION_BY_NAME[aggregation_name],
            )
        if scorer_kind in {'CONTACT_MAP', 'CONTACTMAPSCORER'}:
            return PTContactMapScorer()
        if scorer_kind in {'GENE_MASK_LFC', 'GENEMASKLFCSCORER'}:
            return PTGeneMaskLFCScorer(
                requested_output=_OFFICIAL_TO_PT_OUTPUT[_normalize_output_type(requested_output)],
                resolution=1,
            )
        if scorer_kind in {'GENE_MASK_ACTIVE', 'GENEMASKACTIVESCORER'}:
            return PTGeneMaskActiveScorer(
                requested_output=_OFFICIAL_TO_PT_OUTPUT[_normalize_output_type(requested_output)],
                resolution=1,
            )
        if scorer_kind in {'GENE_MASK_SPLICING', 'GENEMASKSPLICINGSCORER'}:
            return PTGeneMaskSplicingScorer(
                requested_output=_OFFICIAL_TO_PT_OUTPUT[_normalize_output_type(requested_output)],
                width=getattr(scorer, 'width', None),
            )
        if scorer_kind in {'PA_QTL', 'POLYADENYLATIONSCORER'}:
            return PTPolyadenylationScorer()
        if scorer_kind in {'SPLICE_JUNCTION', 'SPLICEJUNCTIONSCORER'}:
            return PTSpliceJunctionScorer()

        # Proto-style scorer descriptor support.
        which = getattr(scorer, 'WhichOneof', None)
        if callable(which):
            field = scorer.WhichOneof('scorer')
            if field == 'center_mask':
                center = scorer.center_mask
                return PTCenterMaskScorer(
                    requested_output=_OFFICIAL_TO_PT_OUTPUT[dna_output.OutputType(center.requested_output)],
                    width=center.width if center.HasField('width') else None,
                    aggregation_type=_PT_AGGREGATION_BY_NAME[
                        dna_model_pb2.AggregationType.Name(center.aggregation_type)
                        .removeprefix('AGGREGATION_TYPE_')
                    ],
                )
            if field == 'contact_map':
                return PTContactMapScorer()
            if field == 'gene_mask':
                return PTGeneMaskLFCScorer(
                    requested_output=_OFFICIAL_TO_PT_OUTPUT[dna_output.OutputType(scorer.gene_mask.requested_output)],
                    resolution=1,
                )
            if field == 'gene_mask_active':
                return PTGeneMaskActiveScorer(
                    requested_output=_OFFICIAL_TO_PT_OUTPUT[
                        dna_output.OutputType(scorer.gene_mask_active.requested_output)
                    ],
                    resolution=1,
                )
            if field == 'gene_mask_splicing':
                return PTGeneMaskSplicingScorer(
                    requested_output=_OFFICIAL_TO_PT_OUTPUT[
                        dna_output.OutputType(scorer.gene_mask_splicing.requested_output)
                    ],
                    width=scorer.gene_mask_splicing.width
                    if scorer.gene_mask_splicing.HasField('width')
                    else None,
                )
            if field == 'pa_qtl':
                return PTPolyadenylationScorer()
            if field == 'splice_junction':
                return PTSpliceJunctionScorer()

        raise ValueError(f'Unsupported variant scorer type: {scorer!r}')

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
                var=pd.DataFrame(columns=['name', 'strand']),
                uns={'interval': interval, 'variant': variant, 'variant_scorer': fallback_variant_scorer},
            )

        x_rows = []
        obs_rows = []
        has_gene_metadata = False
        for i, score in enumerate(score_list):
            values = _as_numpy(score.scores).astype(np.float32, copy=False)
            if values.ndim != 1:
                values = values.reshape(-1)
            x_rows.append(values)

            row = {
                'gene_id': score.gene_id,
                'gene_name': score.gene_name,
                'gene_type': score.gene_type,
                'strand': score.gene_strand,
                'junction_Start': score.junction_start,
                'junction_End': score.junction_end,
            }
            if any(v is not None for v in row.values()):
                has_gene_metadata = True
            obs_rows.append(row)

        X = np.stack(x_rows, axis=0).astype(np.float32, copy=False)
        n_tracks = X.shape[1] if X.ndim == 2 else 0

        scorer_for_output = score_list[0].scorer if score_list else None
        requested_output = getattr(scorer_for_output, 'requested_output', None)
        if isinstance(requested_output, PTOutputType):
            pt_output = requested_output
        else:
            pt_output = _OFFICIAL_TO_PT_OUTPUT.get(
                _normalize_output_type(getattr(fallback_variant_scorer, 'requested_output', dna_output.OutputType.RNA_SEQ)),
                PTOutputType.RNA_SEQ,
            )
        var_df = self._build_track_metadata_df(
            pt_output_type=pt_output,
            organism_index=organism_index,
            num_tracks=n_tracks,
        )

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
                'interval': interval,
                'variant': variant,
                'variant_scorer': fallback_variant_scorer,
            },
        )

    @staticmethod
    def _squeeze_batch(values: np.ndarray) -> np.ndarray:
        if values.ndim > 0 and values.shape[0] == 1:
            return values[0]
        return values

    @staticmethod
    def _ensure_batch(values: np.ndarray) -> np.ndarray:
        if values.ndim > 0 and values.shape[0] != 1:
            return values
        if values.ndim == 0:
            return values[None]
        return values if values.shape[0] == 1 else values[None]


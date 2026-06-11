"""Local serving adapter for AlphaGenome notebook-compatible APIs.

This module provides a local implementation of the notebook-facing AlphaGenome
prediction surface (`predict_*`, `output_metadata`) by wrapping the shared
``AlphaGenomePredictionRuntime`` and converting outputs to upstream-compatible
containers. Variant scoring lives in ``scorer.py``.
"""

from __future__ import annotations

import dataclasses
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

from alphagenome_pytorch.prediction import AlphaGenomePredictionRuntime
from alphagenome_pytorch.extensions.attribution import (
    AttributionResult,
    get_method,
    UnsupportedMethodError,
)
from alphagenome_pytorch.extensions.attribution.heads import default_head_selector
from alphagenome_pytorch.extensions.attribution.window import target_slice_for_resolution
from alphagenome_pytorch.variant_scoring.types import (
    Interval as PTInterval,
    OutputType as PTOutputType,
    TrackMetadata as PTTrackMetadata,
    Variant as PTVariant,
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

DEFAULT_MAX_WORKERS = 5
MAX_VARIANT_SCORERS_PER_REQUEST = 20
VALID_SEQUENCE_CHARACTERS = frozenset('ACGTN')

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

def _validate_sequence_length(length: int) -> None:
    if length not in SUPPORTED_SEQUENCE_LENGTHS.values():
        raise ValueError(
            f'Sequence length {length} not supported. '
            f'Supported lengths: {list(SUPPORTED_SEQUENCE_LENGTHS.values())}'
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
            if normalized in dna_output.OutputType.__members__:
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


def _pt_metadata_to_track_df(
    metadata: Sequence[PTTrackMetadata],
    num_tracks: int | None = None,
) -> pd.DataFrame:
    rows = []
    for i, meta in enumerate(metadata):
        extras_get = getattr(meta, 'get', None)
        get = extras_get if callable(extras_get) else lambda name, default=None: getattr(meta, name, default)
        track_name = getattr(meta, 'track_name', None) or get('name', None)
        track_strand = getattr(meta, 'track_strand', None) or get('strand', '.')
        rows.append(
            {
                'name': track_name or f'track_{i}',
                'strand': track_strand or '.',
                'ontology_curie': get('ontology_curie', None),
                'gtex_tissue': get('gtex_tissue', None),
                'Assay title': get('assay_title', None) or get('Assay title', None),
                'biosample_name': get('biosample_name', None),
                'biosample_type': get('biosample_type', None),
                'transcription_factor': get('transcription_factor', None),
                'histone_mark': get('histone_mark', None),
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
    metadata: Sequence[PTTrackMetadata],
    num_tracks: int | None = None,
) -> pd.DataFrame:
    rows = []
    for i, meta in enumerate(metadata):
        extras_get = getattr(meta, 'get', None)
        get = extras_get if callable(extras_get) else lambda name, default=None: getattr(meta, name, default)
        track_name = getattr(meta, 'track_name', None) or get('name', None)
        rows.append(
            {
                'name': track_name or f'track_{i}',
                'ontology_curie': get('ontology_curie', None),
                'gtex_tissue': get('gtex_tissue', None),
                'Assay title': get('assay_title', None) or get('Assay title', None),
                'biosample_name': get('biosample_name', None),
                'biosample_type': get('biosample_type', None),
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


class LocalDnaModelAdapter:
    """Notebook-compatible local model adapter.

    This class mirrors the notebook-critical subset of the AlphaGenome API:
    `predict_sequence`, `predict_interval`, `predict_variant`, and
    `output_metadata`.
    """

    def __init__(
        self,
        runtime: AlphaGenomePredictionRuntime,
        *,
        scorer: 'VariantScorer | None' = None,
    ):
        self.runtime = runtime
        self.scorer = scorer

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
        organism_index = self.runtime.resolve_organism_index(organism)
        outputs = self.runtime.predict(sequence, organism=organism_index)
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
        sequence = self.runtime.get_sequence(interval)
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
        organism_index = self.runtime.resolve_organism_index(organism)

        ref_outputs, alt_outputs = self.runtime.predict_variant(
            interval=interval,
            variant=variant,
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
        raise NotImplementedError('score_interval is not implemented.')

    def score_variant(self, *args, **kwargs):
        if self.scorer is None:
            raise NotImplementedError('Variant scoring not available for this model.')
        return self.scorer.score_variant(*args, **kwargs)

    def score_variants(self, *args, **kwargs):
        if self.scorer is None:
            raise NotImplementedError('Variant scoring not available for this model.')
        return self.scorer.score_variants(*args, **kwargs)

    def score_ism_variants(self, *args, **kwargs):
        if self.scorer is None:
            raise NotImplementedError('Variant scoring not available for this model.')
        return self.scorer.score_ism_variants(*args, **kwargs)

    def explain_interval(
        self,
        *,
        interval: genome.Interval,
        target_interval: genome.Interval,
        organism: Any = dna_model_pb2.ORGANISM_HOMO_SAPIENS,
        requested_output: str,
        resolution: int,
        track_indices: Sequence[int],
        method: str,
        reduction: str = "sum",
        include_raw_gradient: bool = False,
        strand_averaged: bool = False,
        batch_size: int = 8,
    ) -> 'AttributionResult':
        """Nucleotide attribution over a target window inside an interval.

        This is the serving surface for per-base attribution (gradient × input
        and saturation ISM).  It calls into
        :mod:`alphagenome_pytorch.extensions.attribution` — **not** through
        ``VariantScoringModel.predict()`` — because gradient methods require an
        active autograd graph.

        Args:
            interval: Full input interval (must be a supported sequence length).
            target_interval: Sub-interval to attribute over — must be contained
                in ``interval``.
            organism: Organism identifier (proto enum, string, or int).
            requested_output: Head name, e.g. ``"dnase"``.
            resolution: Output resolution in bp (1 or 128).
            track_indices: Which tracks to attribute.
            method: ``"input_x_gradient"`` or ``"saturation_ism"``.
            reduction: Window reduction (``"sum"``, ``"mean"``, ``"peak"``).
            include_raw_gradient: Return full ``(W, 4, T)`` gradient tensor
                (only valid for gradient-based methods).
            strand_averaged: Average forward and reverse-complement attributions.
            batch_size: Batch size for ISM mutation loop.

        Returns:
            :class:`~alphagenome_pytorch.extensions.attribution.types.AttributionResult`.

        Raises:
            ValueError: On invalid inputs (containment, unknown head, bad indices, …).
            UnsupportedMethodError: If ``method`` is not in the registry.
        """
        # --- validation -------------------------------------------------
        _validate_sequence_length(interval.width)

        if target_interval.chromosome != interval.chromosome:
            raise ValueError(
                f"target_interval chromosome ({target_interval.chromosome}) "
                f"must match interval chromosome ({interval.chromosome})."
            )
        if target_interval.start < interval.start or target_interval.end > interval.end:
            raise ValueError(
                f"target_interval {target_interval.chromosome}:"
                f"{target_interval.start}-{target_interval.end} is not "
                f"contained within interval {interval.chromosome}:"
                f"{interval.start}-{interval.end}."
            )

        # Method lookup (raises UnsupportedMethodError on unknown method).
        method_spec = get_method(method)

        if include_raw_gradient and not method_spec.supports_raw_gradient:
            raise ValueError(
                f"include_raw_gradient is not supported for method {method!r}."
            )

        if not track_indices:
            raise ValueError("track_indices must be non-empty.")

        # --- sequence → one-hot -----------------------------------------
        organism_index = self.runtime.resolve_organism_index(organism)
        sequence = self.runtime.get_sequence(interval)

        from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor

        device = self.runtime.device
        onehot = sequence_to_onehot_tensor(
            sequence, dtype=torch.float32, device=device,
        ).unsqueeze(0)  # (1, L, 4)

        # --- target window bookkeeping -----------------------------------
        target_slice = target_slice_for_resolution(
            interval.start, target_interval.start, target_interval.end, resolution,
        )

        # --- dispatch ----------------------------------------------------
        model = self.runtime.model

        kwargs: dict[str, Any] = dict(
            onehot=onehot,
            organism_index=organism_index,
            output_type=requested_output,
            resolution=resolution,
            target_slice=target_slice,
            track_indices=track_indices,
            reduction=reduction,
            strand_averaged=strand_averaged,
            head_selector=default_head_selector,
            sequence=sequence,
            target_start=target_interval.start,
            target_end=target_interval.end,
        )
        if method_spec.supports_raw_gradient:
            kwargs["include_raw_gradient"] = include_raw_gradient
        if method == "saturation_ism":
            kwargs["batch_size"] = batch_size

        return method_spec.func(model, **kwargs)

    def output_metadata(
        self,
        organism: Any = dna_model_pb2.ORGANISM_HOMO_SAPIENS,
    ) -> dna_output.OutputMetadata:
        organism_index = self.runtime.resolve_organism_index(organism)

        metadata_kwargs: dict[str, pd.DataFrame | None] = {
            field.name: None for field in dataclasses.fields(dna_output.OutputMetadata)
        }
        for official_output_type, field_name in _OFFICIAL_OUTPUT_FIELD.items():
            pt_output_type = _OFFICIAL_TO_PT_OUTPUT[official_output_type]
            entries = self.runtime.get_track_metadata(
                organism_index,
                output_name=pt_output_type.value,
            )
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
            available = ', '.join(sorted(outputs.keys())) or '<none>'
            raise ValueError(
                f'Requested output "{key}" not produced by this model. '
                f'Available outputs: {available}.'
            )
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
        metadata = self.runtime.get_track_metadata(
            organism_index,
            output_name=pt_output_type.value,
        )
        return self._pt_metadata_to_track_df(metadata, num_tracks=num_tracks)

    def _build_junction_metadata_df(
        self,
        *,
        organism_index: int,
        num_tracks: int,
    ) -> pd.DataFrame:
        metadata = self.runtime.get_track_metadata(
            organism_index,
            output_name=PTOutputType.SPLICE_JUNCTIONS.value,
        )
        return self._pt_metadata_to_junction_df(metadata, num_tracks=num_tracks)

    def _pt_metadata_to_track_df(
        self,
        metadata: Sequence[PTTrackMetadata],
        num_tracks: int | None = None,
    ) -> pd.DataFrame:
        return _pt_metadata_to_track_df(metadata, num_tracks=num_tracks)

    def _pt_metadata_to_junction_df(
        self,
        metadata: Sequence[PTTrackMetadata],
        num_tracks: int | None = None,
    ) -> pd.DataFrame:
        return _pt_metadata_to_junction_df(metadata, num_tracks=num_tracks)

    @staticmethod
    def _squeeze_batch(values: np.ndarray) -> np.ndarray:
        if values.ndim > 0 and values.shape[0] == 1:
            return values[0]
        return values

    @staticmethod
    def _ensure_batch(values: np.ndarray) -> np.ndarray:
        """Promote a 0-d scalar to a length-1 array; pass arrays through unchanged.

        This intentionally does **not** wrap arrays whose leading dim is not 1.
        Callers in :py:meth:`_build_junction_data` rely on existing batch
        dimensions being preserved as-is. If you need "wrap any non-1 leading
        dim with a fresh batch axis" semantics, that is a different contract —
        introduce a new helper rather than changing this one.
        """
        return values[None] if values.ndim == 0 else values

"""gRPC transport for local AlphaGenome serving."""

from __future__ import annotations

import dataclasses
import itertools
import logging
from collections.abc import Iterable, Iterator, Sequence
from concurrent import futures
from typing import Any

import grpc
import numpy as np
import pandas as pd

from alphagenome import tensor_utils
from alphagenome.data import genome
from alphagenome.data import junction_data as ag_junction_data
from alphagenome.data import track_data as ag_track_data
from alphagenome.models import dna_output
from alphagenome.models import junction_data_utils, track_data_utils
from alphagenome.protos import dna_model_pb2, dna_model_service_pb2, dna_model_service_pb2_grpc, tensor_pb2

from .adapter import LocalDnaModelAdapter

LOGGER = logging.getLogger(__name__)


def _set_grpc_error(context: grpc.ServicerContext, exc: Exception) -> None:
    if isinstance(exc, ValueError):
        code = grpc.StatusCode.INVALID_ARGUMENT
    elif isinstance(exc, NotImplementedError):
        code = grpc.StatusCode.UNIMPLEMENTED
    else:
        code = grpc.StatusCode.INTERNAL
    context.abort(code, str(exc))


def _first_request(request_iterator: Iterator[Any], method_name: str) -> Any:
    try:
        return next(request_iterator)
    except StopIteration as exc:
        raise ValueError(f'{method_name} expected one request message, got empty stream.') from exc


def _normalize_output_type(value: int) -> dna_output.OutputType:
    return dna_output.OutputType(value)


def _normalize_ontology_terms_from_proto(
    ontology_terms: Sequence[dna_model_pb2.OntologyTerm],
) -> list[str] | None:
    if not ontology_terms:
        return None
    # OntologyTerm proto fields are (ontology_type, id).
    ids = [term.id for term in ontology_terms if term.id]
    return list(dict.fromkeys(ids))


def _metadata_df_to_track_proto(metadata: pd.DataFrame) -> dna_model_pb2.TracksMetadata:
    df = metadata.copy()
    if 'name' not in df.columns:
        df['name'] = [f'track_{i}' for i in range(len(df))]
    if 'strand' not in df.columns:
        df['strand'] = '.'
    return track_data_utils.metadata_to_proto(df)


def _metadata_df_to_junction_proto(metadata: pd.DataFrame) -> dna_model_pb2.JunctionsMetadata:
    df = metadata.copy()
    if 'name' not in df.columns:
        df['name'] = [f'track_{i}' for i in range(len(df))]
    return junction_data_utils.metadata_to_proto(df)


def _as_variant_proto(value: Any) -> dna_model_pb2.Variant | None:
    if value is None:
        return None
    if hasattr(value, 'to_proto'):
        return value.to_proto()
    if isinstance(value, dna_model_pb2.Variant):
        return value
    if isinstance(value, str):
        try:
            return genome.Variant.from_str(value).to_proto()
        except Exception:
            return None
    return None


def _as_gene_metadata(obs: pd.DataFrame | None) -> list[dna_model_pb2.GeneScorerMetadata]:
    if obs is None or obs.empty:
        return []
    rows: list[dna_model_pb2.GeneScorerMetadata] = []
    for _, row in obs.iterrows():
        gene_id = row.get('gene_id', '') if isinstance(row, pd.Series) else ''
        if pd.isna(gene_id):
            gene_id = ''
        metadata = dna_model_pb2.GeneScorerMetadata(gene_id=str(gene_id))

        strand = row.get('strand')
        if isinstance(strand, str) and strand in ('+', '-', '.'):
            metadata.strand = genome.Strand.from_str(strand).to_proto()

        gene_name = row.get('gene_name')
        if isinstance(gene_name, str) and gene_name:
            metadata.name = gene_name

        gene_type = row.get('gene_type')
        if isinstance(gene_type, str) and gene_type:
            metadata.type = gene_type

        junction_start = row.get('junction_Start')
        if junction_start is not None and not pd.isna(junction_start):
            metadata.junction_start = int(junction_start)

        junction_end = row.get('junction_End')
        if junction_end is not None and not pd.isna(junction_end):
            metadata.junction_end = int(junction_end)

        rows.append(metadata)
    return rows


def _anndata_to_score_variant_output(
    adata: Any,
    *,
    bytes_per_chunk: int,
    compression_type: tensor_pb2.CompressionType,
) -> tuple[dna_model_pb2.ScoreVariantOutput, Sequence[tensor_pb2.TensorChunk]]:
    X = np.asarray(adata.X, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]

    layers = getattr(adata, 'layers', None)
    if layers is not None and 'quantiles' in layers:
        quantiles = np.asarray(layers['quantiles'], dtype=np.float32)
        packed_values = np.stack([X, quantiles], axis=0)
    else:
        packed_values = np.expand_dims(X, axis=0)

    tensor, chunks = tensor_utils.pack_tensor(
        packed_values,
        bytes_per_chunk=bytes_per_chunk,
        compression_type=compression_type,
    )

    var = getattr(adata, 'var', pd.DataFrame())
    if not isinstance(var, pd.DataFrame):
        var = pd.DataFrame(var)
    track_metadata = _metadata_df_to_track_proto(var).metadata

    obs = getattr(adata, 'obs', pd.DataFrame())
    if not isinstance(obs, pd.DataFrame):
        obs = pd.DataFrame(obs)
    gene_metadata = _as_gene_metadata(obs)

    uns = getattr(adata, 'uns', {}) or {}
    variant_proto = _as_variant_proto(uns.get('variant'))

    score_output = dna_model_pb2.ScoreVariantOutput(
        variant_data=dna_model_pb2.VariantData(
            values=tensor,
            metadata=dna_model_pb2.VariantMetadata(
                variant=variant_proto,
                track_metadata=track_metadata,
                gene_metadata=gene_metadata,
            ),
        ),
    )
    return score_output, chunks


def _iter_output_payloads(
    output: dna_output.Output,
    *,
    response_cls: type[
        dna_model_service_pb2.PredictSequenceResponse
        | dna_model_service_pb2.PredictIntervalResponse
        | dna_model_service_pb2.PredictVariantResponse
    ],
    output_field_name: str,
    bytes_per_chunk: int,
    compression_type: tensor_pb2.CompressionType,
) -> Iterable[
    dna_model_service_pb2.PredictSequenceResponse
    | dna_model_service_pb2.PredictIntervalResponse
    | dna_model_service_pb2.PredictVariantResponse
]:
    for field in dataclasses.fields(output):
        output_type = field.metadata['output_type']
        value = getattr(output, field.name)
        if value is None:
            continue

        if isinstance(value, ag_track_data.TrackData):
            payload, chunks = track_data_utils.to_protos(
                value,
                bytes_per_chunk=bytes_per_chunk,
                compression_type=compression_type,
            )
            output_proto = dna_model_pb2.Output(
                output_type=output_type.to_proto(),
                track_data=payload,
            )
        elif isinstance(value, ag_junction_data.JunctionData):
            payload, chunks = junction_data_utils.to_protos(
                value,
                bytes_per_chunk=bytes_per_chunk,
                compression_type=compression_type,
            )
            output_proto = dna_model_pb2.Output(
                output_type=output_type.to_proto(),
                junction_data=payload,
            )
        else:
            tensor, chunks = tensor_utils.pack_tensor(
                np.asarray(value),
                bytes_per_chunk=bytes_per_chunk,
                compression_type=compression_type,
            )
            output_proto = dna_model_pb2.Output(
                output_type=output_type.to_proto(),
                data=tensor,
            )

        yield response_cls(**{output_field_name: output_proto})
        for chunk in chunks:
            yield response_cls(tensor_chunk=chunk)


class LocalDnaModelService(dna_model_service_pb2_grpc.DnaModelServiceServicer):
    """gRPC implementation of notebook-critical DNA model methods."""

    def __init__(
        self,
        adapter: LocalDnaModelAdapter,
        *,
        bytes_per_chunk: int = 0,
        compression_type: tensor_pb2.CompressionType = tensor_pb2.COMPRESSION_TYPE_NONE,
    ):
        self.adapter = adapter
        self.bytes_per_chunk = bytes_per_chunk
        self.compression_type = compression_type

    def PredictSequence(self, request_iterator, context):
        try:
            request = _first_request(request_iterator, 'PredictSequence')
            output = self.adapter.predict_sequence(
                sequence=request.sequence,
                organism=request.organism,
                requested_outputs=[_normalize_output_type(v) for v in request.requested_outputs],
                ontology_terms=_normalize_ontology_terms_from_proto(request.ontology_terms),
            )
            yield from _iter_output_payloads(
                output,
                response_cls=dna_model_service_pb2.PredictSequenceResponse,
                output_field_name='output',
                bytes_per_chunk=self.bytes_per_chunk,
                compression_type=self.compression_type,
            )
        except Exception as exc:  # pragma: no cover - exercised in error-path tests.
            _set_grpc_error(context, exc)

    def PredictInterval(self, request_iterator, context):
        try:
            request = _first_request(request_iterator, 'PredictInterval')
            interval = genome.Interval.from_proto(request.interval)
            output = self.adapter.predict_interval(
                interval=interval,
                organism=request.organism,
                requested_outputs=[_normalize_output_type(v) for v in request.requested_outputs],
                ontology_terms=_normalize_ontology_terms_from_proto(request.ontology_terms),
            )
            yield from _iter_output_payloads(
                output,
                response_cls=dna_model_service_pb2.PredictIntervalResponse,
                output_field_name='output',
                bytes_per_chunk=self.bytes_per_chunk,
                compression_type=self.compression_type,
            )
        except Exception as exc:  # pragma: no cover
            _set_grpc_error(context, exc)

    def PredictVariant(self, request_iterator, context):
        try:
            request = _first_request(request_iterator, 'PredictVariant')
            interval = genome.Interval.from_proto(request.interval)
            variant = genome.Variant.from_proto(request.variant)
            output = self.adapter.predict_variant(
                interval=interval,
                variant=variant,
                organism=request.organism,
                requested_outputs=[_normalize_output_type(v) for v in request.requested_outputs],
                ontology_terms=_normalize_ontology_terms_from_proto(request.ontology_terms),
            )
            yield from _iter_output_payloads(
                output.reference,
                response_cls=dna_model_service_pb2.PredictVariantResponse,
                output_field_name='reference_output',
                bytes_per_chunk=self.bytes_per_chunk,
                compression_type=self.compression_type,
            )
            yield from _iter_output_payloads(
                output.alternate,
                response_cls=dna_model_service_pb2.PredictVariantResponse,
                output_field_name='alternate_output',
                bytes_per_chunk=self.bytes_per_chunk,
                compression_type=self.compression_type,
            )
        except Exception as exc:  # pragma: no cover
            _set_grpc_error(context, exc)

    def ScoreInterval(self, request_iterator, context):
        del request_iterator  # Unused.
        context.abort(grpc.StatusCode.UNIMPLEMENTED, 'ScoreInterval is not implemented.')

    def ScoreVariant(self, request_iterator, context):
        try:
            request = _first_request(request_iterator, 'ScoreVariant')
            interval = genome.Interval.from_proto(request.interval)
            variant = genome.Variant.from_proto(request.variant)
            scores = self.adapter.score_variant(
                interval=interval,
                variant=variant,
                variant_scorers=list(request.variant_scorers),
                organism=request.organism,
            )
            for score in scores:
                score_output, chunks = _anndata_to_score_variant_output(
                    score,
                    bytes_per_chunk=self.bytes_per_chunk,
                    compression_type=self.compression_type,
                )
                yield dna_model_service_pb2.ScoreVariantResponse(output=score_output)
                for chunk in chunks:
                    yield dna_model_service_pb2.ScoreVariantResponse(tensor_chunk=chunk)
        except Exception as exc:  # pragma: no cover
            _set_grpc_error(context, exc)

    def ScoreIsmVariant(self, request_iterator, context):
        try:
            request = _first_request(request_iterator, 'ScoreIsmVariant')
            interval = genome.Interval.from_proto(request.interval)
            ism_interval = genome.Interval.from_proto(request.ism_interval)
            interval_variant = (
                genome.Variant.from_proto(request.interval_variant)
                if request.HasField('interval_variant')
                else None
            )
            scores_nested = self.adapter.score_ism_variants(
                interval=interval,
                ism_interval=ism_interval,
                variant_scorers=list(request.variant_scorers),
                organism=request.organism,
                interval_variant=interval_variant,
            )
            for score in itertools.chain.from_iterable(scores_nested):
                score_output, chunks = _anndata_to_score_variant_output(
                    score,
                    bytes_per_chunk=self.bytes_per_chunk,
                    compression_type=self.compression_type,
                )
                yield dna_model_service_pb2.ScoreIsmVariantResponse(output=score_output)
                for chunk in chunks:
                    yield dna_model_service_pb2.ScoreIsmVariantResponse(tensor_chunk=chunk)
        except Exception as exc:  # pragma: no cover
            _set_grpc_error(context, exc)

    def GetMetadata(self, request, context):
        try:
            metadata = self.adapter.output_metadata(request.organism)
            output_metadata = []
            for output_type in dna_output.OutputType:
                data = metadata.get(output_type)
                if data is None:
                    continue
                if output_type == dna_output.OutputType.SPLICE_JUNCTIONS:
                    payload = dna_model_pb2.OutputMetadata(
                        output_type=output_type.to_proto(),
                        junctions=_metadata_df_to_junction_proto(data),
                    )
                else:
                    payload = dna_model_pb2.OutputMetadata(
                        output_type=output_type.to_proto(),
                        tracks=_metadata_df_to_track_proto(data),
                    )
                output_metadata.append(payload)
            yield dna_model_service_pb2.MetadataResponse(output_metadata=output_metadata)
        except Exception as exc:  # pragma: no cover
            _set_grpc_error(context, exc)


def serve_grpc(
    adapter: LocalDnaModelAdapter,
    *,
    host: str = '127.0.0.1',
    port: int = 50051,
    max_workers: int = 8,
    bytes_per_chunk: int = 0,
    compression_type: tensor_pb2.CompressionType = tensor_pb2.COMPRESSION_TYPE_NONE,
    wait: bool = True,
) -> grpc.Server:
    """Start a local gRPC server that implements `DnaModelService`."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', -1),
            ('grpc.max_receive_message_length', -1),
        ],
    )
    dna_model_service_pb2_grpc.add_DnaModelServiceServicer_to_server(
        LocalDnaModelService(
            adapter=adapter,
            bytes_per_chunk=bytes_per_chunk,
            compression_type=compression_type,
        ),
        server,
    )

    bind_address = f'{host}:{port}'
    bound_port = server.add_insecure_port(bind_address)
    if bound_port == 0:
        raise RuntimeError(
            f'Failed to bind gRPC server to {bind_address}; the port may already be in use.'
        )
    server.start()
    LOGGER.info('Local gRPC serving started at %s:%d', host, bound_port)

    if wait:
        server.wait_for_termination()
    return server

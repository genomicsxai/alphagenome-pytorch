from __future__ import annotations

from concurrent import futures

import grpc
import pytest

from alphagenome import tensor_utils
from alphagenome.data import genome
from alphagenome.models import dna_output
from alphagenome.protos import dna_model_pb2, dna_model_service_pb2, dna_model_service_pb2_grpc

from alphagenome_pytorch.extensions.serving.adapter import LocalDnaModelAdapter, SEQUENCE_LENGTH_16KB
from alphagenome_pytorch.extensions.serving.grpc_service import LocalDnaModelService
from alphagenome_pytorch.extensions.serving.scorer import VariantScorer

from .serving_fakes import FakeAnndataModule, FakeRuntime, FakeScoringModel


def _start_grpc(adapter):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    dna_model_service_pb2_grpc.add_DnaModelServiceServicer_to_server(
        LocalDnaModelService(adapter),
        server,
    )
    port = server.add_insecure_port('127.0.0.1:0')
    server.start()
    channel = grpc.insecure_channel(f'127.0.0.1:{port}')
    stub = dna_model_service_pb2_grpc.DnaModelServiceStub(channel)
    try:
        yield stub
    finally:
        channel.close()
        server.stop(grace=0.0)


@pytest.fixture
def grpc_server(monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'anndata', FakeAnndataModule)
    runtime = FakeRuntime()
    adapter = LocalDnaModelAdapter(
        runtime, scorer=VariantScorer(runtime, FakeScoringModel()),
    )
    yield from _start_grpc(adapter)


@pytest.fixture
def catalog_grpc_server(monkeypatch):
    """A router-backed (catalog-mode) service with one entry, ``'m1'``.

    The two catalog error paths (missing / unknown ``alphagenome-model-id``)
    reject before any model computation, so the entry's adapter is never built
    — a bare ``nn.Linear`` base and a lambda adapter factory are enough.
    """
    import torch.nn as nn

    from alphagenome_pytorch.extensions.serving.router import (
        ServedModelEntry,
        ServedModelRouter,
    )

    monkeypatch.setitem(__import__('sys').modules, 'anndata', FakeAnndataModule)
    entry = ServedModelEntry(
        id='m1', label='M1', kind='adapter', base_model_hash='sha256:demo',
    )
    router = ServedModelRouter(
        base_model=nn.Linear(4, 4),
        runtime=FakeRuntime(),
        entries=[entry],
        adapter_factory=lambda r, e: object(),  # unreached on the error paths
    )
    yield from _start_grpc(router)


def _catalog_predict_request():
    return dna_model_service_pb2.PredictSequenceRequest(
        sequence='A' * SEQUENCE_LENGTH_16KB,
        organism=dna_model_pb2.ORGANISM_HOMO_SAPIENS,
        requested_outputs=[dna_output.OutputType.DNASE.to_proto()],
    )


def test_catalog_missing_model_id_is_failed_precondition(catalog_grpc_server):
    """Regression: a deliberate abort raised inside ``_acquire`` must not be
    swallowed by the RPC method's broad ``except`` and re-aborted as INTERNAL."""
    with pytest.raises(grpc.RpcError) as exc_info:
        list(catalog_grpc_server.PredictSequence(iter([_catalog_predict_request()])))
    assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    # Details must survive too — the INTERNAL bug also blanked them.
    assert 'alphagenome-model-id' in exc_info.value.details()


def test_catalog_unknown_model_id_is_not_found(catalog_grpc_server):
    """Regression: unknown model id must surface NOT_FOUND, not INTERNAL."""
    metadata = (('alphagenome-model-id', 'does-not-exist'),)
    with pytest.raises(grpc.RpcError) as exc_info:
        list(
            catalog_grpc_server.PredictSequence(
                iter([_catalog_predict_request()]), metadata=metadata
            )
        )
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND
    assert 'does-not-exist' in exc_info.value.details()


@pytest.fixture
def prediction_only_grpc_server(monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'anndata', FakeAnndataModule)
    adapter = LocalDnaModelAdapter(FakeRuntime())
    yield from _start_grpc(adapter)


def test_predict_sequence_rpc(grpc_server):
    request = dna_model_service_pb2.PredictSequenceRequest(
        sequence='A' * SEQUENCE_LENGTH_16KB,
        organism=dna_model_pb2.ORGANISM_HOMO_SAPIENS,
        requested_outputs=[dna_output.OutputType.DNASE.to_proto()],
    )
    responses = list(grpc_server.PredictSequence(iter([request])))
    assert len(responses) == 1
    assert responses[0].WhichOneof('payload') == 'output'
    assert responses[0].output.output_type == dna_output.OutputType.DNASE.to_proto()
    assert responses[0].output.track_data.values.WhichOneof('payload') == 'array'


def test_score_variant_rpc(grpc_server):
    request = dna_model_service_pb2.ScoreVariantRequest(
        interval=genome.Interval('chr1', 0, SEQUENCE_LENGTH_16KB).to_proto(),
        variant=genome.Variant('chr1', 10, 'A', 'C').to_proto(),
        organism=dna_model_pb2.ORGANISM_HOMO_SAPIENS,
        variant_scorers=[
            dna_model_pb2.VariantScorer(
                center_mask=dna_model_pb2.CenterMaskScorer(
                    requested_output=dna_output.OutputType.DNASE.to_proto(),
                    width=501,
                    aggregation_type=dna_model_pb2.AGGREGATION_TYPE_DIFF_SUM,
                )
            )
        ],
    )
    responses = list(grpc_server.ScoreVariant(iter([request])))
    assert len(responses) == 1
    assert responses[0].WhichOneof('payload') == 'output'
    values = tensor_utils.unpack_proto(responses[0].output.variant_data.values)
    assert values.shape == (1, 1, 2)
    assert responses[0].output.variant_data.metadata.variant.chromosome == 'chr1'


def test_score_variant_rpc_prediction_only_unimplemented(prediction_only_grpc_server):
    request = dna_model_service_pb2.ScoreVariantRequest(
        interval=genome.Interval('chr1', 0, SEQUENCE_LENGTH_16KB).to_proto(),
        variant=genome.Variant('chr1', 10, 'A', 'C').to_proto(),
        organism=dna_model_pb2.ORGANISM_HOMO_SAPIENS,
    )
    with pytest.raises(grpc.RpcError) as exc_info:
        list(prediction_only_grpc_server.ScoreVariant(iter([request])))
    assert exc_info.value.code() == grpc.StatusCode.UNIMPLEMENTED


def test_metadata_rpc(grpc_server):
    request = dna_model_service_pb2.MetadataRequest(
        organism=dna_model_pb2.ORGANISM_HOMO_SAPIENS
    )
    responses = list(grpc_server.GetMetadata(request))
    assert len(responses) == 1
    by_type = {m.output_type for m in responses[0].output_metadata}
    assert dna_output.OutputType.DNASE.to_proto() in by_type


def test_organism_or_none_maps_unspecified_to_none():
    """gRPC omitting organism sends ORGANISM_UNSPECIFIED (0); it must become None
    so the runtime's embedded default (e.g. a mouse bundle) is consulted."""
    from alphagenome.protos import dna_model_pb2

    from alphagenome_pytorch.extensions.serving.grpc_service import (
        _organism_or_none,
    )

    assert _organism_or_none(dna_model_pb2.ORGANISM_UNSPECIFIED) is None
    assert (
        _organism_or_none(dna_model_pb2.ORGANISM_MUS_MUSCULUS)
        == dna_model_pb2.ORGANISM_MUS_MUSCULUS
    )
    assert (
        _organism_or_none(dna_model_pb2.ORGANISM_HOMO_SAPIENS)
        == dna_model_pb2.ORGANISM_HOMO_SAPIENS
    )

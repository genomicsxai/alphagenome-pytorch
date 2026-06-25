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

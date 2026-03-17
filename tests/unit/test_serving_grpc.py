from __future__ import annotations

from concurrent import futures

import grpc
import numpy as np
import pandas as pd
import pytest
import torch

from alphagenome import tensor_utils
from alphagenome.data import genome
from alphagenome.models import dna_output
from alphagenome.protos import dna_model_pb2, dna_model_service_pb2, dna_model_service_pb2_grpc

from alphagenome_pytorch.extensions.serving.adapter import LocalDnaModelAdapter, SEQUENCE_LENGTH_16KB
from alphagenome_pytorch.extensions.serving.grpc_service import LocalDnaModelService
from alphagenome_pytorch.variant_scoring.types import (
    Interval as PTInterval,
    OutputType as PTOutputType,
    TrackMetadata as PTTrackMetadata,
    Variant as PTVariant,
    VariantScore,
)


class _FakeAnndataModule:
    class AnnData:
        def __init__(self, X, obs=None, var=None, uns=None, layers=None):
            self.X = X
            self.obs = obs if obs is not None else pd.DataFrame()
            self.var = var if var is not None else pd.DataFrame()
            self.uns = uns if uns is not None else {}
            self.layers = layers if layers is not None else {}


class _FakeScoringModel:
    def __init__(self):
        self._metadata = {
            0: {
                PTOutputType.DNASE: [
                    PTTrackMetadata(0, 'track_a', '.', PTOutputType.DNASE, ontology_curie='CL:0001'),
                    PTTrackMetadata(1, 'track_b', '.', PTOutputType.DNASE, ontology_curie='CL:0002'),
                ]
            }
        }

    def get_track_metadata(self, organism=None):
        del organism
        return self._metadata[0]

    def predict(self, sequence, organism=None, **kwargs):
        del organism, kwargs
        seq_len = len(sequence)
        values = np.ones((1, seq_len, 2), dtype=np.float32)
        return {'dnase': {1: values}}

    def get_sequence(self, interval, variant=None):
        del variant
        return 'A' * interval.width

    def predict_variant(self, interval, variant, organism=None):
        del variant, organism
        ref = self.predict('A' * interval.width)
        alt = self.predict('A' * interval.width)
        alt['dnase'][1] = alt['dnase'][1] + 1.0
        return ref, alt

    def score_variant(self, interval, variant, scorers, organism=None):
        del interval, variant, organism
        result = []
        for scorer in scorers:
            result.append(
                VariantScore(
                    variant=PTVariant('chr1', 10, 'A', 'C'),
                    interval=PTInterval('chr1', 0, SEQUENCE_LENGTH_16KB),
                    scorer=scorer,
                    scores=torch.tensor([1.0, 2.0]),
                )
            )
        return result


@pytest.fixture
def grpc_server(monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'anndata', _FakeAnndataModule)
    adapter = LocalDnaModelAdapter(_FakeScoringModel())
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


def test_metadata_rpc(grpc_server):
    request = dna_model_service_pb2.MetadataRequest(
        organism=dna_model_pb2.ORGANISM_HOMO_SAPIENS
    )
    responses = list(grpc_server.GetMetadata(request))
    assert len(responses) == 1
    by_type = {m.output_type for m in responses[0].output_metadata}
    assert dna_output.OutputType.DNASE.to_proto() in by_type


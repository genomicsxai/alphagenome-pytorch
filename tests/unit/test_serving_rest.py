from __future__ import annotations

import http.client
import json

import numpy as np
import pandas as pd
import pytest
import torch

from alphagenome.models import dna_output

from alphagenome_pytorch.extensions.serving.adapter import LocalDnaModelAdapter, SEQUENCE_LENGTH_16KB
from alphagenome_pytorch.extensions.serving.rest_service import serve_rest
from alphagenome_pytorch.variant_scoring.scorers import (
    CenterMaskScorer as PTCenterMaskScorer,
    ContactMapScorer as PTContactMapScorer,
    GeneMaskActiveScorer as PTGeneMaskActiveScorer,
    GeneMaskLFCScorer as PTGeneMaskLFCScorer,
    GeneMaskSplicingScorer as PTGeneMaskSplicingScorer,
    PolyadenylationScorer as PTPolyadenylationScorer,
    SpliceJunctionScorer as PTSpliceJunctionScorer,
)
from alphagenome_pytorch.variant_scoring.scorers.gene_mask import GeneMaskMode
from alphagenome_pytorch.variant_scoring.types import (
    AggregationType as PTAggregationType,
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
def rest_server(monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'anndata', _FakeAnndataModule)
    adapter = LocalDnaModelAdapter(_FakeScoringModel())
    server = serve_rest(adapter, host='127.0.0.1', port=0, wait=False)
    host, port = server.server_address
    try:
        yield host, port
    finally:
        server.shutdown()
        server.server_close()


def _post(host: str, port: int, path: str, body: bytes) -> tuple[int, dict]:
    conn = http.client.HTTPConnection(host, port, timeout=10)
    try:
        conn.request('POST', path, body=body, headers={'Content-Type': 'application/json'})
        response = conn.getresponse()
        payload = json.loads(response.read().decode('utf-8'))
        return response.status, payload
    finally:
        conn.close()


def test_predict_sequence_happy_path(rest_server):
    host, port = rest_server
    body = json.dumps({
        'sequence': 'A' * SEQUENCE_LENGTH_16KB,
        'organism': 'HOMO_SAPIENS',
        'requested_outputs': [dna_output.OutputType.DNASE.name],
    }).encode('utf-8')
    status, payload = _post(host, port, '/v1/predict_sequence', body)
    assert status == 200
    assert 'output' in payload
    dnase = payload['output']['dnase']
    assert dnase is not None
    assert dnase['resolution'] == 1
    assert np.asarray(dnase['values']).shape == (SEQUENCE_LENGTH_16KB, 2)
    assert payload['output']['atac'] is None


def test_predict_sequence_malformed_json(rest_server):
    host, port = rest_server
    status, payload = _post(host, port, '/v1/predict_sequence', b'{not json')
    assert status == 400
    assert 'error' in payload


def test_score_variant_with_center_mask_scorer(rest_server):
    host, port = rest_server
    body = json.dumps({
        'interval': {'chromosome': 'chr1', 'start': 0, 'end': SEQUENCE_LENGTH_16KB},
        'variant': {
            'chromosome': 'chr1', 'position': 10,
            'reference_bases': 'A', 'alternate_bases': 'C',
        },
        'variant_scorers': [
            {
                'type': 'center_mask',
                'requested_output': 'DNASE',
                'width': 501,
                'aggregation_type': 'DIFF_SUM',
            }
        ],
        'organism': 'HOMO_SAPIENS',
    }).encode('utf-8')
    status, payload = _post(host, port, '/v1/score_variant', body)
    assert status == 200
    assert len(payload['scores']) == 1
    uns = payload['scores'][0]['uns']
    assert 'CenterMaskScorer' in uns['variant_scorer']


def test_score_variant_unknown_scorer_type(rest_server):
    host, port = rest_server
    body = json.dumps({
        'interval': {'chromosome': 'chr1', 'start': 0, 'end': SEQUENCE_LENGTH_16KB},
        'variant': {
            'chromosome': 'chr1', 'position': 10,
            'reference_bases': 'A', 'alternate_bases': 'C',
        },
        'variant_scorers': [{'type': 'bogus_scorer'}],
    }).encode('utf-8')
    status, payload = _post(host, port, '/v1/score_variant', body)
    assert status == 400
    assert 'bogus_scorer' in payload['error']
    assert 'center_mask' in payload['error']


def test_score_variant_missing_required_field(rest_server):
    host, port = rest_server
    body = json.dumps({
        'interval': {'chromosome': 'chr1', 'start': 0, 'end': SEQUENCE_LENGTH_16KB},
        'variant': {
            'chromosome': 'chr1', 'position': 10,
            'reference_bases': 'A', 'alternate_bases': 'C',
        },
        'variant_scorers': [
            {'type': 'center_mask', 'requested_output': 'DNASE', 'width': 501},
        ],
    }).encode('utf-8')
    status, payload = _post(host, port, '/v1/score_variant', body)
    assert status == 400
    assert 'aggregation_type' in payload['error']


def test_parse_variant_scorers_all_types():
    from alphagenome_pytorch.extensions.serving.rest_service import _parse_variant_scorers

    parsed = _parse_variant_scorers([
        {'type': 'center_mask', 'requested_output': 'DNASE', 'width': 501, 'aggregation_type': 'DIFF_SUM'},
        {'type': 'contact_map'},
        {'type': 'gene_mask_lfc', 'requested_output': 'RNA_SEQ'},
        {'type': 'gene_mask_active', 'requested_output': 'DNASE', 'mask_mode': 'body'},
        {'type': 'gene_mask_splicing', 'requested_output': 'SPLICE_SITES', 'width': 1001},
        {'type': 'polyadenylation', 'min_pas_count': 3, 'min_pas_coverage': 0.9},
        {'type': 'splice_junction', 'filter_protein_coding': False},
    ])
    assert isinstance(parsed[0], PTCenterMaskScorer)
    assert parsed[0].aggregation_type == PTAggregationType.DIFF_SUM
    assert parsed[0].requested_output == PTOutputType.DNASE
    assert isinstance(parsed[1], PTContactMapScorer)
    assert isinstance(parsed[2], PTGeneMaskLFCScorer)
    assert parsed[2].mask_mode == GeneMaskMode.EXONS  # default
    assert isinstance(parsed[3], PTGeneMaskActiveScorer)
    assert parsed[3].mask_mode == GeneMaskMode.BODY  # case-insensitive parse
    assert isinstance(parsed[4], PTGeneMaskSplicingScorer)
    assert parsed[4].width == 1001
    assert isinstance(parsed[5], PTPolyadenylationScorer)
    assert isinstance(parsed[6], PTSpliceJunctionScorer)


def test_parse_variant_scorers_rejects_bad_shape():
    from alphagenome_pytorch.extensions.serving.rest_service import _parse_variant_scorers

    with pytest.raises(ValueError, match='must be a JSON list'):
        _parse_variant_scorers({'type': 'center_mask'})

    with pytest.raises(ValueError, match='must be a JSON object'):
        _parse_variant_scorers(['not a dict'])

    with pytest.raises(ValueError, match='Unknown AggregationType'):
        _parse_variant_scorers([
            {'type': 'center_mask', 'requested_output': 'DNASE',
             'width': 501, 'aggregation_type': 'NOT_A_REAL_AGG'},
        ])


def test_parse_variant_scorers_empty_returns_empty_list():
    from alphagenome_pytorch.extensions.serving.rest_service import _parse_variant_scorers

    assert _parse_variant_scorers(None) == []
    assert _parse_variant_scorers([]) == []

from __future__ import annotations

import http.client
import json

import numpy as np
import pytest

from alphagenome.models import dna_output

from alphagenome_pytorch.extensions.serving.adapter import LocalDnaModelAdapter, SEQUENCE_LENGTH_16KB
from alphagenome_pytorch.extensions.serving.rest_service import serve_rest
from alphagenome_pytorch.extensions.serving.scorer import VariantScorer
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
    OutputType as PTOutputType,
)

from .serving_fakes import FakeAnndataModule, FakeRuntime, FakeScoringModel


def _start_rest(adapter):
    server = serve_rest(adapter, host='127.0.0.1', port=0, wait=False)
    host, port = server.server_address
    try:
        yield host, port
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def rest_server(monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'anndata', FakeAnndataModule)
    runtime = FakeRuntime()
    adapter = LocalDnaModelAdapter(
        runtime, scorer=VariantScorer(runtime, FakeScoringModel()),
    )
    yield from _start_rest(adapter)


@pytest.fixture
def prediction_only_rest_server(monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'anndata', FakeAnndataModule)
    adapter = LocalDnaModelAdapter(FakeRuntime())
    yield from _start_rest(adapter)


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


def test_score_variant_prediction_only_adapter_returns_501(prediction_only_rest_server):
    host, port = prediction_only_rest_server
    body = json.dumps({
        'interval': {'chromosome': 'chr1', 'start': 0, 'end': SEQUENCE_LENGTH_16KB},
        'variant': {
            'chromosome': 'chr1', 'position': 10,
            'reference_bases': 'A', 'alternate_bases': 'C',
        },
        'variant_scorers': [],
    }).encode('utf-8')
    status, payload = _post(host, port, '/v1/score_variant', body)
    assert status == 501
    assert 'Variant scoring not available' in payload['error']


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


# ---------------------------------------------------------------------------
# /v1/explain_interval tests
# ---------------------------------------------------------------------------


def test_explain_interval_gradient_round_trip(rest_server):
    host, port = rest_server
    body = json.dumps({
        'interval': {'chromosome': 'chr1', 'start': 0, 'end': SEQUENCE_LENGTH_16KB},
        'target_interval': {'chromosome': 'chr1', 'start': 100, 'end': 200},
        'organism': 'HOMO_SAPIENS',
        'requested_output': 'dnase',
        'resolution': 1,
        'track_indices': [0],
        'method': 'input_x_gradient',
        'reduction': 'sum',
    }).encode('utf-8')
    status, payload = _post(host, port, '/v1/explain_interval', body)
    assert status == 200
    assert 'attribution' in payload
    attr = payload['attribution']
    assert attr['method'] == 'input_x_gradient'
    assert attr['kind'] == 'base_matrix'
    # values shape should be (100, 4, 1) encoded as nested lists
    values = np.asarray(attr['values'])
    assert values.shape == (100, 4, 1)
    assert attr['target_start'] == 100
    assert attr['target_end'] == 200
    assert attr['raw_gradient'] is None


def test_explain_interval_ism_round_trip(rest_server):
    host, port = rest_server
    body = json.dumps({
        'interval': {'chromosome': 'chr1', 'start': 0, 'end': SEQUENCE_LENGTH_16KB},
        'target_interval': {'chromosome': 'chr1', 'start': 100, 'end': 108},
        'requested_output': 'dnase',
        'resolution': 1,
        'track_indices': [0],
        'method': 'saturation_ism',
        'batch_size': 4,
    }).encode('utf-8')
    status, payload = _post(host, port, '/v1/explain_interval', body)
    assert status == 200
    attr = payload['attribution']
    assert attr['method'] == 'saturation_ism'
    # ISM reference-base cells are NaN → serialized as null
    values = attr['values']  # nested list
    # Check that at least one cell is null (the reference base for each position)
    flat = json.dumps(values)
    assert 'null' in flat, 'Reference-base cells should serialize as null'


def test_explain_interval_unknown_method_400(rest_server):
    host, port = rest_server
    body = json.dumps({
        'interval': {'chromosome': 'chr1', 'start': 0, 'end': SEQUENCE_LENGTH_16KB},
        'target_interval': {'chromosome': 'chr1', 'start': 100, 'end': 200},
        'requested_output': 'dnase',
        'resolution': 1,
        'track_indices': [0],
        'method': 'nonexistent_method',
    }).encode('utf-8')
    status, payload = _post(host, port, '/v1/explain_interval', body)
    assert status == 400
    assert 'nonexistent_method' in payload['error']
    # Error should list known methods
    assert 'input_x_gradient' in payload['error']
    assert 'saturation_ism' in payload['error']

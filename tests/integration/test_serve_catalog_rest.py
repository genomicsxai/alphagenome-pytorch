"""REST catalog-mode integration test.

Spins up an HTTP server backed by a :class:`ServedModelRouter` over two
fake adapters and exercises the catalog routing rules end-to-end:
``GET /v1/models``, scoped metadata, optional ``model_id`` in POST bodies,
ambiguity / unknown-id error responses.

Uses the shared :mod:`tests.unit.serving_fakes` so we don't need a real
``AlphaGenome`` model.
"""

from __future__ import annotations

import http.client
import json
from typing import Any

import pytest
import torch.nn as nn

from alphagenome.models import dna_output  # noqa: F401  (test depends on alphagenome)

from alphagenome_pytorch.extensions.serving.adapter import (
    LocalDnaModelAdapter,
    SEQUENCE_LENGTH_16KB,
)
from alphagenome_pytorch.extensions.serving.rest_service import serve_rest
from alphagenome_pytorch.extensions.serving.router import (
    ServedModelEntry,
    ServedModelRouter,
)

from tests.unit.serving_fakes import FakeAnndataModule, FakeRuntime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_entry(eid: str, runtime: FakeRuntime) -> tuple[ServedModelEntry, LocalDnaModelAdapter]:
    """Build a pre-resolved entry + its bound adapter for testing."""
    entry = ServedModelEntry(
        id=eid, label=f"label-{eid}", kind="adapter",
        base_model_hash="sha256:demo",
    )
    return entry, LocalDnaModelAdapter(runtime)


def _make_router(*, ids: list[str]) -> tuple[ServedModelRouter, dict[str, FakeRuntime]]:
    runtimes: dict[str, FakeRuntime] = {eid: FakeRuntime() for eid in ids}
    entries = [
        ServedModelEntry(
            id=eid, label=f"label-{eid}", kind="adapter",
            base_model_hash="sha256:demo",
        )
        for eid in ids
    ]

    # Attach a tagged value to each runtime's predict so requests can be
    # distinguished in assertions.
    for i, eid in enumerate(ids):
        rt = runtimes[eid]
        original_predict = rt.predict

        def _tagged(seq, organism=None, _rt=rt, _idx=i, **kw):
            out = original_predict(seq, organism=organism, **kw)
            # Encode the model index into the data so we can verify routing.
            out['dnase'][1][:] = _idx + 1
            return out
        rt.predict = _tagged  # type: ignore[assignment]

    base_model = nn.Module()  # router only uses identity, not forward
    base_model.heads = nn.ModuleDict()  # type: ignore[attr-defined]

    def factory(_router, entry):
        return LocalDnaModelAdapter(runtimes[entry.id])

    router = ServedModelRouter(
        base_model=base_model,
        runtime=runtimes[ids[0]],  # any runtime works as the shared "base" runtime
        entries=entries,
        adapter_factory=factory,
    )
    return router, runtimes


@pytest.fixture
def two_model_server(monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'anndata', FakeAnndataModule)
    router, _runtimes = _make_router(ids=["alpha", "beta"])
    server = serve_rest(router, host='127.0.0.1', port=0, wait=False)
    host, port = server.server_address
    try:
        yield host, port
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def single_model_server(monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'anndata', FakeAnndataModule)
    router, _runtimes = _make_router(ids=["only"])
    server = serve_rest(router, host='127.0.0.1', port=0, wait=False)
    host, port = server.server_address
    try:
        yield host, port
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(host: str, port: int, path: str) -> tuple[int, dict]:
    conn = http.client.HTTPConnection(host, port, timeout=10)
    try:
        conn.request('GET', path)
        response = conn.getresponse()
        payload = json.loads(response.read().decode('utf-8'))
        return response.status, payload
    finally:
        conn.close()


def _post(host: str, port: int, path: str, body: dict) -> tuple[int, dict]:
    conn = http.client.HTTPConnection(host, port, timeout=10)
    try:
        conn.request(
            'POST', path,
            body=json.dumps(body).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode('utf-8'))
        return response.status, payload
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_list_models_two_entries(two_model_server):
    host, port = two_model_server
    status, payload = _get(host, port, '/v1/models')
    assert status == 200
    ids = [m['id'] for m in payload['models']]
    assert ids == ['alpha', 'beta']
    assert all(m['base_model_hash'] == 'sha256:demo' for m in payload['models'])


def test_list_models_singleton_mode(monkeypatch):
    """Singleton mode should still respond to /v1/models with a degenerate row."""
    monkeypatch.setitem(__import__('sys').modules, 'anndata', FakeAnndataModule)
    adapter = LocalDnaModelAdapter(FakeRuntime())
    server = serve_rest(adapter, host='127.0.0.1', port=0, wait=False)
    host, port = server.server_address
    try:
        status, payload = _get(host, port, '/v1/models')
        assert status == 200
        assert payload['models'][0]['kind'] == 'singleton'
    finally:
        server.shutdown()
        server.server_close()


def test_predict_routes_via_model_id(two_model_server):
    host, port = two_model_server
    body_alpha = {
        'sequence': 'A' * SEQUENCE_LENGTH_16KB,
        'organism': 'HOMO_SAPIENS',
        'requested_outputs': ['DNASE'],
        'model_id': 'alpha',
    }
    status, payload = _post(host, port, '/v1/predict_sequence', body_alpha)
    assert status == 200
    # alpha is index 0 → tagged value 1
    assert payload['output']['dnase']['values'][0][0] == 1.0

    body_beta = dict(body_alpha, model_id='beta')
    status, payload = _post(host, port, '/v1/predict_sequence', body_beta)
    assert status == 200
    # beta is index 1 → tagged value 2
    assert payload['output']['dnase']['values'][0][0] == 2.0


def test_scoped_route(two_model_server):
    host, port = two_model_server
    body = {
        'sequence': 'A' * SEQUENCE_LENGTH_16KB,
        'organism': 'HOMO_SAPIENS',
        'requested_outputs': ['DNASE'],
    }
    status, payload = _post(host, port, '/v1/models/beta/predict_sequence', body)
    assert status == 200
    assert payload['output']['dnase']['values'][0][0] == 2.0


def test_scoped_metadata_route(two_model_server):
    host, port = two_model_server
    status, payload = _get(host, port, '/v1/models/alpha/metadata')
    assert status == 200
    assert 'metadata' in payload


def test_missing_model_id_is_400(two_model_server):
    host, port = two_model_server
    body = {
        'sequence': 'A' * SEQUENCE_LENGTH_16KB,
        'organism': 'HOMO_SAPIENS',
        'requested_outputs': ['DNASE'],
    }
    status, payload = _post(host, port, '/v1/predict_sequence', body)
    assert status == 400
    assert 'model_id' in payload['error'].lower() or 'specify' in payload['error'].lower()


def test_unknown_model_id_is_404(two_model_server):
    host, port = two_model_server
    body = {
        'sequence': 'A' * SEQUENCE_LENGTH_16KB,
        'organism': 'HOMO_SAPIENS',
        'requested_outputs': ['DNASE'],
        'model_id': 'does-not-exist',
    }
    status, payload = _post(host, port, '/v1/predict_sequence', body)
    assert status == 404
    assert 'does-not-exist' in payload['error']


def test_single_model_catalog_omits_model_id(single_model_server):
    """With one model registered, missing model_id picks the only entry."""
    host, port = single_model_server
    body = {
        'sequence': 'A' * SEQUENCE_LENGTH_16KB,
        'organism': 'HOMO_SAPIENS',
        'requested_outputs': ['DNASE'],
    }
    status, payload = _post(host, port, '/v1/predict_sequence', body)
    assert status == 200
    assert payload['output']['dnase']['values'][0][0] == 1.0


def test_url_body_model_id_mismatch_is_400(two_model_server):
    host, port = two_model_server
    body = {
        'sequence': 'A' * SEQUENCE_LENGTH_16KB,
        'organism': 'HOMO_SAPIENS',
        'requested_outputs': ['DNASE'],
        'model_id': 'beta',
    }
    status, payload = _post(host, port, '/v1/models/alpha/predict_sequence', body)
    assert status == 400
    assert 'mismatch' in payload['error'].lower()

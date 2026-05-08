"""Integration test: upstream DnaClient (API) vs local served DnaClient.

This test validates that the local gRPC serving extension can be consumed by
the official ``alphagenome.models.dna_client.DnaClient`` and that score outputs
are directionally consistent with hosted API results for a representative scorer.
"""

from __future__ import annotations

import collections
import importlib
import os
from concurrent import futures
from pathlib import Path

import numpy as np
import pytest

grpc = pytest.importorskip("grpc")

from .api_utils import (
    DEFAULT_VARIANT_ALTERNATE_BASES,
    DEFAULT_VARIANT_CHROMOSOME,
    DEFAULT_VARIANT_POSITION,
    DEFAULT_VARIANT_REFERENCE_BASES,
)

TEST_SEQUENCE_LENGTH = "16KB"
DEFAULT_LOCAL_RPC_TIMEOUT_SECONDS = float(
    os.environ.get("ALPHAGENOME_LOCAL_RPC_TIMEOUT_SECONDS", "180")
)


class _ClientCallDetails(
    collections.namedtuple(
        "_ClientCallDetails",
        (
            "method",
            "timeout",
            "metadata",
            "credentials",
            "wait_for_ready",
            "compression",
        ),
    ),
    grpc.ClientCallDetails,
):
    pass


class _DefaultStreamStreamTimeoutInterceptor(grpc.StreamStreamClientInterceptor):
    """Injects a default deadline for stream-stream RPC calls."""

    def __init__(self, timeout_seconds: float):
        self._timeout_seconds = timeout_seconds

    def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
        timeout = (
            client_call_details.timeout
            if client_call_details.timeout is not None
            else self._timeout_seconds
        )
        new_details = _ClientCallDetails(
            method=client_call_details.method,
            timeout=timeout,
            metadata=client_call_details.metadata,
            credentials=getattr(client_call_details, "credentials", None),
            wait_for_ready=getattr(client_call_details, "wait_for_ready", None),
            compression=getattr(client_call_details, "compression", None),
        )
        return continuation(new_details, request_iterator)


def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 0.0:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(np.dot(x, y) / denom)


def _flatten_comparable_arrays(remote_values: np.ndarray, local_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    remote = np.asarray(remote_values, dtype=np.float32).reshape(-1)
    local = np.asarray(local_values, dtype=np.float32).reshape(-1)
    assert remote.shape == local.shape
    assert remote.size > 0

    finite_mask = np.isfinite(remote) & np.isfinite(local)
    assert finite_mask.any(), "No finite values available for parity comparison."
    return remote[finite_mask], local[finite_mask]


def _assert_prediction_parity(
    remote_values: np.ndarray,
    local_values: np.ndarray,
    *,
    label: str,
    cosine_threshold: float = 0.70,
    max_normalized_mae: float = 1.50,
) -> None:
    remote, local = _flatten_comparable_arrays(remote_values, local_values)

    cosine = _cosine_similarity(remote, local)
    assert cosine >= cosine_threshold, f"{label}: low API/local cosine similarity: {cosine:.4f}"

    scale = float(np.mean(np.abs(remote)))
    if scale > 1e-6:
        normalized_mae = float(np.mean(np.abs(remote - local)) / scale)
        assert (
            normalized_mae <= max_normalized_mae
        ), f"{label}: high normalized MAE ({normalized_mae:.4f} > {max_normalized_mae:.4f})"


def _dnase_values(output: object) -> np.ndarray:
    dnase = getattr(output, "dnase", None)
    assert dnase is not None, "DNASE output is missing from response."
    values = getattr(dnase, "values", None)
    assert values is not None, "DNASE values are missing from response."
    return np.asarray(values, dtype=np.float32)


@pytest.fixture(scope="module")
def alphagenome_modules(alphagenome_api_key, fasta_path):
    del alphagenome_api_key, fasta_path
    pytest.importorskip("alphagenome")
    return {
        "genome": importlib.import_module("alphagenome.data.genome"),
        "dna_client": importlib.import_module("alphagenome.models.dna_client"),
        "dna_output": importlib.import_module("alphagenome.models.dna_output"),
        "variant_scorers": importlib.import_module("alphagenome.models.variant_scorers"),
        "dna_model_service_pb2_grpc": importlib.import_module(
            "alphagenome.protos.dna_model_service_pb2_grpc"
        ),
    }


@pytest.fixture(scope="module")
def alphagenome_api_key() -> str:
    api_key = os.environ.get("ALPHAGENOME_API_KEY")
    if not api_key:
        pytest.skip("ALPHAGENOME_API_KEY is not set; skipping API parity test.")
    return api_key


@pytest.fixture(scope="module")
def fasta_path() -> str:
    path = os.environ.get("ALPHAGENOME_FASTA_PATH")
    if not path:
        pytest.skip("ALPHAGENOME_FASTA_PATH is not set; skipping API parity test.")
    if not Path(path).exists():
        pytest.skip(f"FASTA file not found at ALPHAGENOME_FASTA_PATH={path}")
    return path


@pytest.fixture(scope="module")
def track_metadata_path() -> str | None:
    path = os.environ.get("ALPHAGENOME_TRACK_METADATA_PATH")
    if path and Path(path).exists():
        return path
    return None


@pytest.fixture(scope="module")
def local_scoring_model(pytorch_model, fasta_path, track_metadata_path):
    VariantScoringModel = importlib.import_module(
        "alphagenome_pytorch.variant_scoring.inference"
    ).VariantScoringModel
    scoring_model = VariantScoringModel(
        model=pytorch_model,
        fasta_path=fasta_path,
        default_organism="human",
    )
    if track_metadata_path is not None:
        scoring_model.load_all_metadata(track_metadata_path)
    return scoring_model


@pytest.fixture(scope="module")
def local_grpc_address(alphagenome_modules, local_scoring_model) -> str:
    LocalDnaModelAdapter = importlib.import_module(
        "alphagenome_pytorch.extensions.serving.adapter"
    ).LocalDnaModelAdapter
    LocalDnaModelService = importlib.import_module(
        "alphagenome_pytorch.extensions.serving.grpc_service"
    ).LocalDnaModelService

    adapter = LocalDnaModelAdapter(local_scoring_model)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )
    dna_model_service_pb2_grpc = alphagenome_modules["dna_model_service_pb2_grpc"]
    dna_model_service_pb2_grpc.add_DnaModelServiceServicer_to_server(
        LocalDnaModelService(adapter),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    if port == 0:
        pytest.skip("Failed to bind ephemeral local gRPC port.")
    server.start()

    try:
        yield f"127.0.0.1:{port}"
    finally:
        server.stop(grace=0.0)


@pytest.fixture(scope="module")
def remote_dna_client(alphagenome_modules, alphagenome_api_key):
    dna_client = alphagenome_modules["dna_client"]
    client = dna_client.create(api_key=alphagenome_api_key, timeout=60.0)
    try:
        yield client
    finally:
        client._channel.close()


@pytest.fixture(scope="module")
def local_dna_client(alphagenome_modules, local_grpc_address):
    dna_client = alphagenome_modules["dna_client"]
    base_channel = grpc.insecure_channel(
        local_grpc_address,
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )
    grpc.channel_ready_future(base_channel).result(timeout=30.0)
    channel = grpc.intercept_channel(
        base_channel,
        _DefaultStreamStreamTimeoutInterceptor(DEFAULT_LOCAL_RPC_TIMEOUT_SECONDS),
    )
    client = dna_client.DnaClient(channel=channel)
    try:
        yield client
    finally:
        channel.close()
        base_channel.close()


@pytest.fixture(scope="module")
def test_variant_and_interval(alphagenome_modules):
    genome = alphagenome_modules["genome"]
    dna_client = alphagenome_modules["dna_client"]
    variant = genome.Variant(
        chromosome=DEFAULT_VARIANT_CHROMOSOME,
        position=DEFAULT_VARIANT_POSITION,
        reference_bases=DEFAULT_VARIANT_REFERENCE_BASES,
        alternate_bases=DEFAULT_VARIANT_ALTERNATE_BASES,
    )
    sequence_length = dna_client.SUPPORTED_SEQUENCE_LENGTHS[
        f"SEQUENCE_LENGTH_{TEST_SEQUENCE_LENGTH}"
    ]
    interval = variant.reference_interval.resize(sequence_length)
    return variant, interval


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.slow
def test_predict_sequence_dnaclient_api_vs_local(
    alphagenome_modules,
    remote_dna_client,
    local_dna_client,
):
    """Compares `predict_sequence` outputs for a single output head."""
    dna_output = alphagenome_modules["dna_output"]
    dna_model = importlib.import_module("alphagenome.models.dna_model")
    dna_client = alphagenome_modules["dna_client"]

    sequence_length = dna_client.SUPPORTED_SEQUENCE_LENGTHS[
        f"SEQUENCE_LENGTH_{TEST_SEQUENCE_LENGTH}"
    ]
    sequence = ("ACGT" * ((sequence_length + 3) // 4))[:sequence_length]

    remote_output = remote_dna_client.predict_sequence(
        sequence=sequence,
        organism=dna_model.Organism.HOMO_SAPIENS,
        requested_outputs=[dna_output.OutputType.DNASE],
        ontology_terms=None,
    )
    local_output = local_dna_client.predict_sequence(
        sequence=sequence,
        organism=dna_model.Organism.HOMO_SAPIENS,
        requested_outputs=[dna_output.OutputType.DNASE],
        ontology_terms=None,
    )

    _assert_prediction_parity(
        _dnase_values(remote_output),
        _dnase_values(local_output),
        label="predict_sequence.dnase",
    )


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.slow
def test_predict_interval_dnaclient_api_vs_local(
    alphagenome_modules,
    remote_dna_client,
    local_dna_client,
    test_variant_and_interval,
):
    """Compares `predict_interval` outputs for a single output head."""
    dna_output = alphagenome_modules["dna_output"]
    dna_model = importlib.import_module("alphagenome.models.dna_model")
    _, interval = test_variant_and_interval

    remote_output = remote_dna_client.predict_interval(
        interval=interval,
        organism=dna_model.Organism.HOMO_SAPIENS,
        requested_outputs=[dna_output.OutputType.DNASE],
        ontology_terms=None,
    )
    local_output = local_dna_client.predict_interval(
        interval=interval,
        organism=dna_model.Organism.HOMO_SAPIENS,
        requested_outputs=[dna_output.OutputType.DNASE],
        ontology_terms=None,
    )

    _assert_prediction_parity(
        _dnase_values(remote_output),
        _dnase_values(local_output),
        label="predict_interval.dnase",
    )


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.slow
def test_predict_variant_dnaclient_api_vs_local(
    alphagenome_modules,
    remote_dna_client,
    local_dna_client,
    test_variant_and_interval,
):
    """Compares `predict_variant` outputs for ref/alt DNASE tracks."""
    dna_output = alphagenome_modules["dna_output"]
    dna_model = importlib.import_module("alphagenome.models.dna_model")
    variant, interval = test_variant_and_interval

    remote_output = remote_dna_client.predict_variant(
        interval=interval,
        variant=variant,
        organism=dna_model.Organism.HOMO_SAPIENS,
        requested_outputs=[dna_output.OutputType.DNASE],
        ontology_terms=None,
    )
    local_output = local_dna_client.predict_variant(
        interval=interval,
        variant=variant,
        organism=dna_model.Organism.HOMO_SAPIENS,
        requested_outputs=[dna_output.OutputType.DNASE],
        ontology_terms=None,
    )

    remote_ref = _dnase_values(remote_output.reference)
    remote_alt = _dnase_values(remote_output.alternate)
    local_ref = _dnase_values(local_output.reference)
    local_alt = _dnase_values(local_output.alternate)

    _assert_prediction_parity(remote_ref, local_ref, label="predict_variant.reference.dnase")
    _assert_prediction_parity(remote_alt, local_alt, label="predict_variant.alternate.dnase")
    _assert_prediction_parity(
        remote_alt - remote_ref,
        local_alt - local_ref,
        label="predict_variant.delta.dnase",
        cosine_threshold=0.50,
        max_normalized_mae=2.50,
    )


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.slow
def test_score_variant_dnaclient_api_vs_local(
    alphagenome_modules,
    remote_dna_client,
    local_dna_client,
):
    """Compares one representative scorer output from API and local serving."""
    genome = alphagenome_modules["genome"]
    dna_client = alphagenome_modules["dna_client"]
    dna_output = alphagenome_modules["dna_output"]
    variant_scorers = alphagenome_modules["variant_scorers"]

    variant = genome.Variant(
        chromosome=DEFAULT_VARIANT_CHROMOSOME,
        position=DEFAULT_VARIANT_POSITION,
        reference_bases=DEFAULT_VARIANT_REFERENCE_BASES,
        alternate_bases=DEFAULT_VARIANT_ALTERNATE_BASES,
    )
    sequence_length = dna_client.SUPPORTED_SEQUENCE_LENGTHS[
        f"SEQUENCE_LENGTH_{TEST_SEQUENCE_LENGTH}"
    ]
    interval = variant.reference_interval.resize(sequence_length)
    scorer = variant_scorers.CenterMaskScorer(
        requested_output=dna_output.OutputType.DNASE,
        width=501,
        aggregation_type=variant_scorers.AggregationType.DIFF_MEAN,
    )

    remote_scores = remote_dna_client.score_variant(
        interval=interval,
        variant=variant,
        variant_scorers=[scorer],
    )
    local_scores = local_dna_client.score_variant(
        interval=interval,
        variant=variant,
        variant_scorers=[scorer],
    )

    assert len(remote_scores) == 1
    assert len(local_scores) == 1

    remote_values = np.asarray(remote_scores[0].X, dtype=np.float32).reshape(-1)
    local_values = np.asarray(local_scores[0].X, dtype=np.float32).reshape(-1)
    assert remote_values.shape == local_values.shape
    assert remote_values.size > 0

    finite_mask = np.isfinite(remote_values) & np.isfinite(local_values)
    assert finite_mask.any(), "No finite values available for parity comparison."
    remote_values = remote_values[finite_mask]
    local_values = local_values[finite_mask]

    cosine = _cosine_similarity(remote_values, local_values)
    assert cosine >= 0.80, f"Low API/local cosine similarity: {cosine:.4f}"

    top_k = min(200, remote_values.size)
    remote_top = set(np.argpartition(np.abs(remote_values), -top_k)[-top_k:])
    local_top = set(np.argpartition(np.abs(local_values), -top_k)[-top_k:])
    overlap = len(remote_top & local_top) / float(top_k)
    assert overlap >= 0.50, f"Low API/local top-{top_k} overlap: {overlap:.4f}"

from __future__ import annotations

import numpy as np
import pytest

from alphagenome.data import genome
from alphagenome.models import dna_output
from alphagenome.protos import dna_model_pb2

from alphagenome_pytorch.extensions.serving.adapter import LocalDnaModelAdapter, SEQUENCE_LENGTH_16KB
from alphagenome_pytorch.extensions.serving.variant_scoring_adapter import VariantScoringAdapter
from alphagenome_pytorch.variant_scoring.scorers import CenterMaskScorer
from alphagenome_pytorch.variant_scoring.types import (
    AggregationType as PTAggregationType,
    OutputType as PTOutputType,
)

from .serving_fakes import FakeAnndataModule, FakeRuntime, FakeScoringModel


@pytest.fixture
def adapter():
    return LocalDnaModelAdapter(FakeRuntime())


@pytest.fixture
def scoring_adapter():
    return VariantScoringAdapter(FakeRuntime(), FakeScoringModel())


def test_predict_sequence_filters_ontology_and_preserves_track_ops(adapter):
    sequence = 'A' * SEQUENCE_LENGTH_16KB
    output = adapter.predict_sequence(
        sequence=sequence,
        requested_outputs={dna_output.OutputType.DNASE},
        ontology_terms=['CL:0001'],
    )

    assert output.dnase is not None
    assert output.dnase.values.shape == (SEQUENCE_LENGTH_16KB, 1)
    assert output.dnase.metadata['ontology_curie'].tolist() == ['CL:0001']
    negative = output.dnase.filter_to_negative_strand()
    assert negative.values.shape[-1] == 0


def test_predict_variant_returns_reference_and_alternate(adapter):
    interval = genome.Interval('chr1', 0, SEQUENCE_LENGTH_16KB)
    variant = genome.Variant('chr1', 10, 'A', 'C')
    output = adapter.predict_variant(
        interval=interval,
        variant=variant,
        requested_outputs=[dna_output.OutputType.DNASE],
        ontology_terms=None,
    )

    assert output.reference.dnase is not None
    assert output.alternate.dnase is not None
    diff = output.alternate.dnase - output.reference.dnase
    assert np.allclose(diff.values, 1.0)


def test_output_metadata_concatenate_has_output_type(adapter):
    metadata = adapter.output_metadata(dna_model_pb2.ORGANISM_HOMO_SAPIENS)
    concatenated = metadata.concatenate()
    assert 'output_type' in concatenated.columns
    assert any(concatenated['output_type'] == dna_output.OutputType.DNASE)


def test_score_variant_returns_anndata_compatible_shape(scoring_adapter, monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'anndata', FakeAnndataModule)

    interval = genome.Interval('chr1', 0, SEQUENCE_LENGTH_16KB)
    variant = genome.Variant('chr1', 10, 'A', 'C')
    scorer = CenterMaskScorer(
        requested_output=PTOutputType.DNASE,
        width=501,
        aggregation_type=PTAggregationType.DIFF_SUM,
    )

    scores = scoring_adapter.score_variant(
        interval=interval,
        variant=variant,
        variant_scorers=[scorer],
        organism=dna_model_pb2.ORGANISM_HOMO_SAPIENS,
    )
    assert len(scores) == 1
    adata = scores[0]
    assert adata.X.shape == (1, 2)
    assert adata.obs.loc['0', 'gene_name'] == 'GENE1'
    assert adata.uns['variant'] == variant
    assert adata.uns['interval'] == interval

from __future__ import annotations

import numpy as np
import pytest

from alphagenome.data import genome
from alphagenome.models import dna_output
from alphagenome.protos import dna_model_pb2

from alphagenome_pytorch.extensions.attribution import UnsupportedMethodError
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


# ---------------------------------------------------------------------------
# explain_interval tests
# ---------------------------------------------------------------------------


class TestExplainIntervalValidation:
    """explain_interval request-validation tests."""

    def test_target_interval_not_contained_raises(self, adapter):
        interval = genome.Interval('chr1', 0, SEQUENCE_LENGTH_16KB)
        target = genome.Interval('chr1', SEQUENCE_LENGTH_16KB - 10, SEQUENCE_LENGTH_16KB + 10)
        with pytest.raises(ValueError, match='not contained'):
            adapter.explain_interval(
                interval=interval,
                target_interval=target,
                requested_output='dnase',
                resolution=1,
                track_indices=[0],
                method='input_x_gradient',
            )

    def test_chromosome_mismatch_raises(self, adapter):
        interval = genome.Interval('chr1', 0, SEQUENCE_LENGTH_16KB)
        target = genome.Interval('chr2', 100, 200)
        with pytest.raises(ValueError, match='chromosome'):
            adapter.explain_interval(
                interval=interval,
                target_interval=target,
                requested_output='dnase',
                resolution=1,
                track_indices=[0],
                method='input_x_gradient',
            )

    def test_unknown_method_raises(self, adapter):
        interval = genome.Interval('chr1', 0, SEQUENCE_LENGTH_16KB)
        target = genome.Interval('chr1', 100, 200)
        with pytest.raises(UnsupportedMethodError, match='bogus_method'):
            adapter.explain_interval(
                interval=interval,
                target_interval=target,
                requested_output='dnase',
                resolution=1,
                track_indices=[0],
                method='bogus_method',
            )

    def test_include_raw_gradient_rejected_for_ism(self, adapter):
        interval = genome.Interval('chr1', 0, SEQUENCE_LENGTH_16KB)
        target = genome.Interval('chr1', 100, 200)
        with pytest.raises(ValueError, match='include_raw_gradient'):
            adapter.explain_interval(
                interval=interval,
                target_interval=target,
                requested_output='dnase',
                resolution=1,
                track_indices=[0],
                method='saturation_ism',
                include_raw_gradient=True,
            )

    def test_empty_track_indices_raises(self, adapter):
        interval = genome.Interval('chr1', 0, SEQUENCE_LENGTH_16KB)
        target = genome.Interval('chr1', 100, 200)
        with pytest.raises(ValueError, match='track_indices'):
            adapter.explain_interval(
                interval=interval,
                target_interval=target,
                requested_output='dnase',
                resolution=1,
                track_indices=[],
                method='input_x_gradient',
            )

    def test_gradient_happy_path(self, adapter):
        interval = genome.Interval('chr1', 0, SEQUENCE_LENGTH_16KB)
        target = genome.Interval('chr1', 100, 200)
        result = adapter.explain_interval(
            interval=interval,
            target_interval=target,
            requested_output='dnase',
            resolution=1,
            track_indices=[0],
            method='input_x_gradient',
        )
        assert result.method == 'input_x_gradient'
        assert result.values.shape == (100, 4, 1)

    def test_ism_happy_path(self, adapter):
        interval = genome.Interval('chr1', 0, SEQUENCE_LENGTH_16KB)
        target = genome.Interval('chr1', 100, 108)
        result = adapter.explain_interval(
            interval=interval,
            target_interval=target,
            requested_output='dnase',
            resolution=1,
            track_indices=[0],
            method='saturation_ism',
            batch_size=4,
        )
        assert result.method == 'saturation_ism'
        assert result.values.shape == (8, 4, 1)

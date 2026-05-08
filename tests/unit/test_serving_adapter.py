from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
import torch

from alphagenome.data import genome
from alphagenome.models import dna_output
from alphagenome.protos import dna_model_pb2

from alphagenome_pytorch.extensions.serving.adapter import LocalDnaModelAdapter, SEQUENCE_LENGTH_16KB
from alphagenome_pytorch.variant_scoring.scorers import CenterMaskScorer
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
                    PTTrackMetadata(
                        track_index=0,
                        track_name='track_a',
                        track_strand='.',
                        output_type=PTOutputType.DNASE,
                        ontology_curie='CL:0001',
                    ),
                    PTTrackMetadata(
                        track_index=1,
                        track_name='track_b',
                        track_strand='-',
                        output_type=PTOutputType.DNASE,
                        ontology_curie='CL:0002',
                    ),
                ]
            }
        }

    def get_track_metadata(self, organism: int | None = None):
        idx = 0 if organism is None else int(organism)
        return self._metadata.get(idx, {})

    def predict(self, sequence: str, organism: int | None = None, **kwargs):
        del organism, kwargs
        seq_len = len(sequence)
        values = np.zeros((1, seq_len, 2), dtype=np.float32)
        values[0, :, 0] = 1.0
        values[0, :, 1] = 2.0
        return {'dnase': {1: values}}

    def get_sequence(self, interval: PTInterval, variant: PTVariant | None = None) -> str:
        seq = list('A' * interval.width)
        if variant is not None:
            rel = variant.start - interval.start
            if 0 <= rel < len(seq):
                seq[rel] = variant.alternate_bases
        return ''.join(seq)

    def predict_variant(self, interval: PTInterval, variant: PTVariant, organism: int | None = None):
        del variant, organism
        ref = self.predict('A' * interval.width)
        alt = self.predict('A' * interval.width)
        alt['dnase'][1] = alt['dnase'][1] + 1.0
        return ref, alt

    def score_variant(self, interval: PTInterval, variant: PTVariant, scorers, organism: int | None = None):
        del interval, variant, organism
        outputs = []
        for scorer in scorers:
            outputs.append(
                VariantScore(
                    variant=PTVariant('chr1', 10, 'A', 'C'),
                    interval=PTInterval('chr1', 0, SEQUENCE_LENGTH_16KB),
                    scorer=scorer,
                    scores=torch.tensor([0.25, -0.5]),
                    gene_id='ENSG000001',
                    gene_name='GENE1',
                    gene_strand='+',
                )
            )
        return outputs


@pytest.fixture
def adapter():
    return LocalDnaModelAdapter(_FakeScoringModel())


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


def test_score_variant_returns_anndata_compatible_shape(adapter, monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'anndata', _FakeAnndataModule)

    interval = genome.Interval('chr1', 0, SEQUENCE_LENGTH_16KB)
    variant = genome.Variant('chr1', 10, 'A', 'C')
    scorer = CenterMaskScorer(
        requested_output=PTOutputType.DNASE,
        width=501,
        aggregation_type=PTAggregationType.DIFF_SUM,
    )

    scores = adapter.score_variant(
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


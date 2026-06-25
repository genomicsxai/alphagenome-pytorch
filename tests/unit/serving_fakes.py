"""Shared test fakes for serving-adapter tests.

Centralises the mocks that were previously duplicated across
``test_serving_adapter.py``, ``test_serving_rest.py`` and
``test_serving_grpc.py``.

Keep these fakes minimal: they expose just enough surface for the adapter
layer to call them. The real behaviours under test live in the adapter,
runtime and REST/gRPC layers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from alphagenome_pytorch.extensions.serving.adapter import SEQUENCE_LENGTH_16KB
from alphagenome_pytorch.variant_scoring.types import (
    Interval as PTInterval,
    OutputType as PTOutputType,
    TrackMetadata as PTTrackMetadata,
    Variant as PTVariant,
    VariantScore,
)


class FakeAnndataModule:
    """Stand-in for the ``anndata`` module so tests don't need it installed."""

    class AnnData:
        def __init__(self, X, obs=None, var=None, uns=None, layers=None):
            self.X = X
            self.obs = obs if obs is not None else pd.DataFrame()
            self.var = var if var is not None else pd.DataFrame()
            self.uns = uns if uns is not None else {}
            self.layers = layers if layers is not None else {}


def _default_metadata():
    """Two DNase tracks with distinct ontology curies and strands.

    ``track_b`` uses ``'-'`` so adapter tests can exercise
    ``filter_to_negative_strand``; the REST/gRPC tests don't depend on the
    strand value.
    """
    return {
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


class FakeScoringModel:
    """Mimics ``VariantScoringModel`` for the scoring code path."""

    def __init__(self):
        self._metadata = _default_metadata()

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

    def score_variant(
        self,
        interval: PTInterval,
        variant: PTVariant,
        scorers,
        organism: int | None = None,
        interval_variant: PTVariant | None = None,
    ):
        del interval, variant, organism, interval_variant
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


class TinyAttributionModel(nn.Module):
    """Minimal differentiable model for attribution tests.

    Returns ``weight * sum_channels(input)`` as a single-track output for the
    ``dnase`` head at resolution 1. Just enough surface for
    ``explain_interval`` / gradient×input / saturation ISM to run.
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def forward(
        self,
        dna_sequence,
        organism_index,
        *,
        heads=None,
        resolutions=None,
        channels_last=True,
        return_scaled_predictions=False,
        **kwargs,
    ):
        del organism_index, heads, resolutions, channels_last
        del return_scaled_predictions, kwargs
        val = self.weight * dna_sequence.sum(dim=-1, keepdim=True)
        return {'dnase': {1: val}}


class FakeRuntime:
    """Mimics ``AlphaGenomePredictionRuntime`` for prediction-only paths."""

    def __init__(self):
        self._metadata = _default_metadata()
        # Mirrors the real runtime's catalog/legacy split; tests can opt in by
        # assigning a catalog instance after construction.
        self.metadata_catalog = None
        self.default_organism_index = 0
        # Carry a tiny differentiable model + device so adapter tests that
        # exercise attribution can dispatch through the runtime without
        # building a separate fixture.
        self.model = TinyAttributionModel()
        self.device = torch.device('cpu')

    def resolve_organism_index(self, organism=None):
        """Match :py:meth:`AlphaGenomePredictionRuntime.resolve_organism_index`.

        Tests pass values like ``9606`` (proto enum value), ``'HOMO_SAPIENS'``,
        or ``'human'``; they should all map to internal index 0. Anything else
        falls through via ``int(organism)``.
        """
        if organism is None:
            return self.default_organism_index
        # Proto-enum NCBI taxonomy IDs.
        if organism == 9606 or organism == 'HOMO_SAPIENS' or organism == 'human':
            return 0
        if organism == 10090 or organism == 'MUS_MUSCULUS' or organism == 'mouse':
            return 1
        return int(organism)

    def predict(self, sequence: str, organism=None, **kwargs):
        del organism, kwargs
        seq_len = len(sequence)
        values = np.zeros((1, seq_len, 2), dtype=np.float32)
        values[0, :, 0] = 1.0
        values[0, :, 1] = 2.0
        return {'dnase': {1: values}}

    def get_sequence(self, interval, variant=None):
        seq = list('A' * interval.width)
        if variant is not None:
            rel = variant.start - interval.start
            if 0 <= rel < len(seq):
                seq[rel] = variant.alternate_bases
        return ''.join(seq)

    def predict_variant(self, interval, variant, organism=None):
        del variant, organism
        ref = self.predict('A' * interval.width)
        alt = self.predict('A' * interval.width)
        alt['dnase'][1] = alt['dnase'][1] + 1.0
        return ref, alt

    def get_track_metadata(self, organism=None, output_name=None):
        idx = 0 if organism is None else int(organism)
        metadata = self._metadata.get(idx, {})
        if output_name is None:
            return metadata
        for key, value in metadata.items():
            if key.value == output_name:
                return value
        return []

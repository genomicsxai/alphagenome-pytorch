from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from alphagenome_pytorch.genome import GenomeSequenceSource
from alphagenome_pytorch.named_outputs import TrackMetadata
from alphagenome_pytorch.prediction import AlphaGenomePredictionRuntime


class _TinyModel(torch.nn.Module):
    num_organisms = 2

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, onehot, organism_index, **kwargs):
        del organism_index, kwargs
        dnase = onehot[..., :1] + 1.0 + self.anchor
        return {"dnase": {1: dnase}}


def _write_fasta(path):
    path.write_text(">chr1\nAAAAAA\n")


def test_prediction_runtime_predict_sequence():
    runtime = AlphaGenomePredictionRuntime(_TinyModel())

    outputs = runtime.predict("AAAA", organism="human")

    assert outputs["dnase"][1].shape == (1, 4, 1)
    assert torch.allclose(outputs["dnase"][1], torch.full((1, 4, 1), 2.0))


def test_prediction_runtime_predict_variant(tmp_path):
    fasta_path = tmp_path / "genome.fa"
    _write_fasta(fasta_path)
    runtime = AlphaGenomePredictionRuntime(
        _TinyModel(),
        sequence_source=GenomeSequenceSource(fasta_path),
    )

    from alphagenome.data import genome

    interval = genome.Interval("chr1", 0, 6)
    variant = genome.Variant("chr1", 2, "A", "C")
    ref, alt = runtime.predict_variant(interval, variant)

    assert ref["dnase"][1][0, 1, 0].item() == 2.0
    assert alt["dnase"][1][0, 1, 0].item() == 1.0


# --- get_track_metadata dispatch tests --------------------------------------
#
# The runtime resolves track metadata through three branches, in order:
#   1. metadata_catalog.get_tracks(output_name, organism=...)   — if catalog present and output_name given
#   2. _legacy_track_metadata[organism_index][output_name]      — dict fallback
#   3. synthesised TrackMetadata from _track_names_for_output() — last-resort placeholder
# These tests pin each dispatch branch.


class _FakeOutputType:
    """Stand-in for a ``PTOutputType``-like enum entry the legacy dict keys on."""

    def __init__(self, name: str):
        self.value = name

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return isinstance(other, _FakeOutputType) and self.value == other.value


def test_get_track_metadata_with_catalog_dispatches_to_catalog():
    catalog = MagicMock()
    catalog.get_tracks.return_value = ('TRACK_A', 'TRACK_B')

    runtime = AlphaGenomePredictionRuntime(
        _TinyModel(),
        metadata_catalog=catalog,
    )

    result = runtime.get_track_metadata(organism='human', output_name='dnase')

    catalog.get_tracks.assert_called_once_with('dnase', organism=0)
    assert result == ['TRACK_A', 'TRACK_B']


def test_get_track_metadata_legacy_fallback_when_no_catalog():
    dnase_key = _FakeOutputType('dnase')
    legacy = {0: {dnase_key: ['legacy_track_0', 'legacy_track_1']}}

    runtime = AlphaGenomePredictionRuntime(
        _TinyModel(),
        track_metadata=legacy,
    )

    by_output_name = runtime.get_track_metadata(organism='human', output_name='dnase')
    assert by_output_name == ['legacy_track_0', 'legacy_track_1']

    # When ``output_name`` is omitted the whole organism dict is returned.
    full = runtime.get_track_metadata(organism='human')
    assert full == {dnase_key: ['legacy_track_0', 'legacy_track_1']}


def test_get_track_metadata_synthesises_from_track_names():
    runtime = AlphaGenomePredictionRuntime(
        _TinyModel(),
        track_names={'dnase': ['synthetic_a', 'synthetic_b']},
    )

    result = runtime.get_track_metadata(organism='human', output_name='dnase')

    assert len(result) == 2
    assert all(isinstance(t, TrackMetadata) for t in result)
    assert [t.track_name for t in result] == ['synthetic_a', 'synthetic_b']
    assert [t.track_index for t in result] == [0, 1]
    assert all(t.organism == 0 and t.output_name == 'dnase' for t in result)


def test_get_track_metadata_returns_empty_when_unknown():
    runtime = AlphaGenomePredictionRuntime(_TinyModel())

    result = runtime.get_track_metadata(organism='human', output_name='dnase')

    assert result == []

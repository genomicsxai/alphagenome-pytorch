"""Adding junction scoring must preserve splice-site scorer coordinates."""

import pytest
import torch

from alphagenome_pytorch.variant_scoring import (
    GeneMaskSplicingScorer,
    Interval,
    OutputType,
    SpliceJunctionScorer,
    Variant,
    VariantScoringModel,
)
from alphagenome_pytorch.variant_scoring.aggregations import align_alternate
from alphagenome_pytorch.utils import splicing


class _RecordingJunctionHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = []

    def forward(self, embedding, organism_index, splice_site_positions):
        self.embeddings.append(embedding.clone())
        count = splice_site_positions.shape[-1]
        return {
            "pred_counts": torch.ones(1, count, count, 2),
            "splice_site_positions": splice_site_positions,
        }


class _DummyModel(torch.nn.Module):
    num_organisms = 2

    def __init__(self):
        super().__init__()
        self.splice_sites_junction_head = _RecordingJunctionHead()


class _FakeFasta:
    def extract(self, interval):
        return "A" * interval.width


@pytest.fixture
def splice_case(monkeypatch):
    """Exercise the real inference/scorers with deterministic mock predictions."""
    wrapper = VariantScoringModel(_DummyModel())
    wrapper._fasta_extractor = _FakeFasta()
    interval = Interval("chr1", 0, 1024)
    generator = torch.Generator().manual_seed(17)
    probabilities = torch.softmax(torch.randn(1, 1024, 5, generator=generator), -1)
    embedding = torch.randn(1, 1024, 4, generator=generator)

    def predict(sequence, organism=None, **kwargs):
        outputs = {"splice_sites": {"probs": probabilities.clone()}}
        if kwargs.get("return_embeddings"):
            outputs.update(
                embeddings_1bp=embedding.clone(),
                embeddings_128bp=torch.zeros(1, 8, 4),
                embeddings_pair=torch.zeros(1, 1, 1, 4),
            )
        return outputs

    wrapper.predict = predict
    position_inputs = []

    def generate_positions(ref, alt, true_splice_sites, **kwargs):
        position_inputs.append((ref.clone(), alt.clone()))
        return torch.tensor([[[490, 510], [495, 520], [490, 510], [495, 520]]])

    monkeypatch.setattr(splicing, "generate_splice_site_positions", generate_positions)
    return wrapper, interval, probabilities, embedding, position_inputs


@pytest.mark.unit
@pytest.mark.parametrize("ref,alt", [("A", "C"), ("A", "AAAC"), ("AAAA", "A")])
@pytest.mark.parametrize("cached_reference", [False, True])
def test_junction_pass_preserves_native_probs_and_aligns_private_inputs(
    splice_case, ref, alt, cached_reference
):
    wrapper, interval, native_probs, native_embedding, position_inputs = splice_case
    variant = Variant("chr1", 500, ref, alt)
    cache = wrapper.predict("A" * interval.width, return_embeddings=True) if cached_reference else None

    reference, alternate = wrapper.predict_variant(
        interval, variant, unified_splicing=True, ref_outputs=cache
    )

    # Public predictions remain in the same native coordinate space regardless
    # of which other scorer requested a junction second pass.
    torch.testing.assert_close(alternate["splice_sites"]["probs"], native_probs, rtol=0, atol=0)
    torch.testing.assert_close(reference["splice_sites"]["probs"], native_probs, rtol=0, atol=0)

    # The fix must still align the private junction-position probabilities and
    # the embeddings supplied to the junction head (including deletion zeros).
    expected_probs = align_alternate(native_probs, variant.start, len(ref), len(alt), interval.start)
    expected_embedding = align_alternate(native_embedding, variant.start, len(ref), len(alt), interval.start)
    assert len(position_inputs) == 1
    torch.testing.assert_close(position_inputs[0][1], expected_probs, rtol=0, atol=0)
    junction_inputs = wrapper.model.splice_sites_junction_head.embeddings
    torch.testing.assert_close(junction_inputs[0], native_embedding.transpose(1, 2), rtol=0, atol=0)
    torch.testing.assert_close(junction_inputs[1], expected_embedding.transpose(1, 2), rtol=0, atol=0)
    if cached_reference:
        assert "embeddings_1bp" in cache
        torch.testing.assert_close(cache["splice_sites"]["probs"], native_probs, rtol=0, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize("ref,alt", [("A", "C"), ("A", "AAAC"), ("AAAA", "A")])
def test_splice_site_score_is_independent_of_junction_scorer(splice_case, ref, alt):
    wrapper, interval, _, _, _ = splice_case
    variant = Variant("chr1", 500, ref, alt)
    sites = GeneMaskSplicingScorer(OutputType.SPLICE_SITES, width=101)

    alone = wrapper.score_variant(interval, variant, [sites])
    together = wrapper.score_variant(interval, variant, [sites, SpliceJunctionScorer()])

    torch.testing.assert_close(alone[0][0].scores, together[0][0].scores, rtol=0, atol=0)

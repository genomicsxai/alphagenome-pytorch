"""Unit tests for variant scoring inference utilities."""

import pandas as pd
import pytest
import torch

from alphagenome_pytorch.variant_scoring import OutputType, VariantScoringModel
from alphagenome_pytorch.variant_scoring.sequence import apply_variant_to_sequence
from alphagenome_pytorch.variant_scoring.types import (
    Interval,
    Variant,
    VariantScore,
)


class _DummyModel(torch.nn.Module):
    num_organisms = 2


class _FakeFasta:
    """Minimal FASTA extractor returning a fixed sequence for any interval."""

    def __init__(self, sequence: str):
        self.sequence = sequence

    def extract(self, interval) -> str:
        return self.sequence


def _seq_value(sequence: str) -> float:
    """Deterministic per-sequence scalar so ref/alt predictions are reproducible."""
    return float(sum(ord(c) for c in sequence) % 9973)


@pytest.mark.unit
def test_load_all_metadata_accepts_legacy_track_strand(monkeypatch, tmp_path):
    metadata_path = tmp_path / "metadata.parquet"
    metadata_path.touch()
    df = pd.DataFrame(
        [
            {
                "organism": "human",
                "output_type": "atac",
                "track_name": "legacy_track",
                "track_strand": "+",
            },
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda path: df)

    scoring_model = VariantScoringModel(_DummyModel())
    scoring_model.load_all_metadata(metadata_path)

    tracks = scoring_model.get_track_metadata("human")[OutputType.ATAC]
    assert tracks[0].track_name == "legacy_track"
    assert tracks[0].track_strand == "+"


@pytest.mark.unit
def test_load_all_metadata_prefers_strand_and_falls_back_for_missing_values(monkeypatch, tmp_path):
    metadata_path = tmp_path / "metadata.parquet"
    metadata_path.touch()
    df = pd.DataFrame(
        [
            {
                "organism": "human",
                "output_type": "atac",
                "track_name": "new_track",
                "strand": "-",
                "track_strand": "+",
            },
            {
                "organism": "human",
                "output_type": "atac",
                "track_name": "fallback_track",
                "strand": pd.NA,
                "track_strand": "+",
            },
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda path: df)

    scoring_model = VariantScoringModel(_DummyModel())
    scoring_model.load_all_metadata(metadata_path)

    tracks = scoring_model.get_track_metadata("human")[OutputType.ATAC]
    assert [track.track_strand for track in tracks] == ["-", "+"]


class _FakeScorer:
    """Duck-typed scorer reading a single fake head; score = alt - ref."""

    name = "FakeScorer()"
    required_heads = frozenset({"atac"})

    def score(self, ref_outputs, alt_outputs, variant, interval,
              organism_index, **kwargs):
        return VariantScore(
            variant=variant,
            interval=interval,
            scorer=self,
            scores=alt_outputs["atac"] - ref_outputs["atac"],
        )


@pytest.mark.unit
def test_score_ism_variants_caches_reference():
    """The reference forward pass runs once for the whole ISM window, and the
    cached-reference scores match recomputing each variant from scratch."""
    ref_seq = "ACGTACGTACGT"
    interval = Interval("chr1", 0, len(ref_seq))

    scoring_model = VariantScoringModel(_DummyModel())
    scoring_model._fasta_extractor = _FakeFasta(ref_seq)

    calls: list[str] = []

    def fake_predict(sequence, organism=None, **kwargs):
        calls.append(sequence)
        return {"atac": torch.tensor(_seq_value(sequence))}

    scoring_model.predict = fake_predict

    scorer = _FakeScorer()
    results = scoring_model.score_ism_variants(
        interval=interval,
        center_position=6,
        scorers=[scorer],
        window_size=3,
        organism=0,
        progress=False,
    )

    # 3 positions x 3 alternate bases = 9 variants.
    assert len(results) == 9
    # Reference forwarded exactly once; one alt forward per variant.
    assert calls.count(ref_seq) == 1
    assert len(calls) == 1 + 9

    # Scores equal alt - ref recomputed independently for each variant — proving
    # the cached reference is the genuine reference and the right alt was scored.
    ref_value = _seq_value(ref_seq)
    for inner in results:
        variant_score = inner[0]
        alt_seq = apply_variant_to_sequence(
            ref_seq, variant_score.variant, interval
        )
        expected = _seq_value(alt_seq) - ref_value
        assert torch.allclose(
            variant_score.scores, torch.tensor(expected, dtype=torch.float32)
        )


@pytest.mark.unit
def test_score_ism_variants_applies_interval_variant_background():
    """A background ``interval_variant`` makes the reference the modified
    sequence, and each ISM SNV is applied on top of it — including at the
    background variant's own position, which used to raise a ref-mismatch."""
    ref_seq = "ACGTACGTACGT"
    interval = Interval("chr1", 0, len(ref_seq))

    scoring_model = VariantScoringModel(_DummyModel())
    scoring_model._fasta_extractor = _FakeFasta(ref_seq)

    calls: list[str] = []

    def fake_predict(sequence, organism=None, **kwargs):
        calls.append(sequence)
        return {"atac": torch.tensor(_seq_value(sequence))}

    scoring_model.predict = fake_predict

    # Background SNV inside the ISM window: 1-based pos 6 -> 0-based idx 5,
    # ref_seq[5] == 'C', flipped to 'A'. The window (center 6, size 3) covers
    # 1-based positions 5, 6, 7 — so it overlaps the background position.
    background = Variant("chr1", 6, "C", "A")
    bg_seq = apply_variant_to_sequence(ref_seq, background, interval)
    assert bg_seq != ref_seq

    results = scoring_model.score_ism_variants(
        interval=interval,
        center_position=6,
        scorers=[_FakeScorer()],
        window_size=3,
        interval_variant=background,
        organism=0,
        progress=False,
    )

    # 3 positions x 3 alts, no crash at the overlapping background position.
    assert len(results) == 9
    # The reference forward pass runs first and uses the background-modified
    # sequence — not the raw reference. (An alt can coincide with the raw
    # reference, e.g. the SNV that reverts the background base, so we pin the
    # reference by position rather than by absence of the raw string.)
    assert calls[0] == bg_seq
    assert calls.count(bg_seq) == 1
    assert len(calls) == 1 + 9

    # Each alt = the background-modified reference with the single ISM SNV on top.
    ref_value = _seq_value(bg_seq)
    for inner in results:
        variant_score = inner[0]
        alt_seq = apply_variant_to_sequence(bg_seq, variant_score.variant, interval)
        expected = _seq_value(alt_seq) - ref_value
        assert torch.allclose(
            variant_score.scores, torch.tensor(expected, dtype=torch.float32)
        )


@pytest.mark.unit
def test_score_ism_variants_rejects_length_changing_background():
    """A length-changing background (indel) would shift ISM coordinates, so it
    is rejected with a clear error rather than silently misaligning."""
    ref_seq = "ACGTACGTACGT"
    interval = Interval("chr1", 0, len(ref_seq))

    scoring_model = VariantScoringModel(_DummyModel())
    scoring_model._fasta_extractor = _FakeFasta(ref_seq)
    scoring_model.predict = lambda *a, **k: {"atac": torch.tensor(0.0)}

    insertion = Variant("chr1", 6, "C", "CC")
    with pytest.raises(ValueError, match="length-preserving"):
        scoring_model.score_ism_variants(
            interval=interval,
            center_position=6,
            scorers=[_FakeScorer()],
            window_size=3,
            interval_variant=insertion,
            organism=0,
            progress=False,
        )


class _FakeSpliceScorer:
    """Duck-typed scorer that triggers the unified-splicing second pass."""

    name = "SpliceJunctionScorer()"
    required_heads = frozenset({"splice_sites"})

    def score(self, ref_outputs, alt_outputs, variant, interval,
              organism_index, **kwargs):
        return VariantScore(
            variant=variant,
            interval=interval,
            scorer=self,
            scores=(alt_outputs["splice_junctions"]
                    - ref_outputs["splice_junctions"]).sum(),
        )


@pytest.mark.unit
def test_score_ism_variants_preserves_reference_cache_with_unified_splicing(
    monkeypatch,
):
    """The unified-splicing pass re-runs the junction head per variant and pops
    embeddings from the working ref dict. The shared cache must survive: if it
    were mutated in place, the second variant's junction pass would fail with
    'Embeddings missing from first pass'. Completing the loop proves the
    shallow-copy protection holds."""
    ref_seq = "ACGTACGTACGT"
    interval = Interval("chr1", 0, len(ref_seq))

    model = _DummyModel()
    model.splice_sites_junction_head = (
        lambda emb_ncl, org_idx, splice_site_positions=None: torch.ones(1, 2)
    )

    scoring_model = VariantScoringModel(model)
    scoring_model._fasta_extractor = _FakeFasta(ref_seq)

    monkeypatch.setattr(
        "alphagenome_pytorch.utils.splicing.generate_splice_site_positions",
        lambda *a, **k: torch.zeros(1, 512, dtype=torch.long),
    )

    calls: list[str] = []

    def fake_predict(sequence, organism=None, **kwargs):
        calls.append(sequence)
        # Fresh dicts per call so popping embeddings from one never affects others.
        return {
            "splice_sites": {"probs": torch.zeros(1, 4, 2)},
            "embeddings_1bp": torch.zeros(1, 4, 3),
        }

    scoring_model.predict = fake_predict

    results = scoring_model.score_ism_variants(
        interval=interval,
        center_position=6,
        scorers=[_FakeSpliceScorer()],
        window_size=3,
        organism=0,
        progress=False,
    )

    assert len(results) == 9
    # Reference forwarded once despite the per-variant junction re-runs.
    assert calls.count(ref_seq) == 1
    assert len(calls) == 1 + 9


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["plain", "splicing"])
def test_score_ism_variants_cached_matches_uncached(monkeypatch, mode):
    """The cached ISM path must produce identical scores to scoring each variant
    independently with no shared reference — i.e. the pre-change behavior, which
    recomputed the reference for every variant. This is the direct old-vs-new
    equivalence check, exercising the real cached/uncached code branches rather
    than a hand-computed expected."""
    ref_seq = "ACGTACGTACGT"
    interval = Interval("chr1", 0, len(ref_seq))

    model = _DummyModel()
    if mode == "splicing":
        # Junction output reflects the embeddings so ref/alt differ per variant.
        model.splice_sites_junction_head = (
            lambda emb_ncl, org_idx, splice_site_positions=None:
            emb_ncl.sum().reshape(1, 1)
        )

    scoring_model = VariantScoringModel(model)
    scoring_model._fasta_extractor = _FakeFasta(ref_seq)

    if mode == "splicing":
        monkeypatch.setattr(
            "alphagenome_pytorch.utils.splicing.generate_splice_site_positions",
            lambda *a, **k: torch.zeros(1, 512, dtype=torch.long),
        )
        scorer = _FakeSpliceScorer()

        def fake_predict(sequence, organism=None, **kwargs):
            value = _seq_value(sequence)
            return {
                "splice_sites": {"probs": torch.zeros(1, 4, 2)},
                "embeddings_1bp": torch.full((1, 4, 3), value),
            }
    else:
        scorer = _FakeScorer()

        def fake_predict(sequence, organism=None, **kwargs):
            return {"atac": torch.tensor(_seq_value(sequence))}

    scoring_model.predict = fake_predict

    cached = scoring_model.score_ism_variants(
        interval=interval,
        center_position=6,
        scorers=[scorer],
        window_size=3,
        organism=0,
        progress=False,
    )

    # Uncached == pre-change behavior: each variant scored on its own, with the
    # reference recomputed (ref_outputs left at its default None).
    for inner in cached:
        cached_score = inner[0]
        uncached = scoring_model.score_variant(
            interval, cached_score.variant, [scorer], organism=0,
        )
        assert torch.allclose(uncached[0].scores, cached_score.scores)

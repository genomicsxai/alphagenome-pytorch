"""Unit tests for VariantScoringModel.score_variant() head filtering (B2.1).

Verifies that score_variant unions each scorer's required_heads and forwards
them to the model forward pass via predict_variant -> predict.
"""

from __future__ import annotations

import pytest
import torch

from alphagenome_pytorch.variant_scoring import (
    AggregationType,
    CenterMaskScorer,
    ContactMapScorer,
    GeneMaskActiveScorer,
    GeneMaskLFCScorer,
    GeneMaskSplicingScorer,
    Interval,
    OutputType,
    PolyadenylationScorer,
    SpliceJunctionScorer,
    Variant,
    VariantScoringModel,
)


class _RecordingModel(torch.nn.Module):
    """Mock model that records the heads kwarg passed to forward()."""

    num_organisms = 2

    def __init__(self):
        super().__init__()
        self.last_heads_seen: tuple[str, ...] | None = None
        self.heads_history: list[tuple[str, ...] | None] = []

        # Tiny dtype_policy stub so .predict() can read compute_dtype.
        class _DP:
            compute_dtype = torch.float32
        self.dtype_policy = _DP()

    def forward(self, dna, organism_index, **kwargs):
        h = kwargs.get('heads')
        self.last_heads_seen = h
        self.heads_history.append(h)
        # Return shape that center mask scorer can consume (B, S, T) at 128bp
        # We only need the requested key to exist.
        B = dna.shape[0]
        S128 = dna.shape[1] // 128
        T = 4
        out: dict = {}
        # Always return atac so center-mask works in tests that ask for it
        out['atac'] = {128: torch.zeros(B, S128, T), 1: torch.zeros(B, dna.shape[1], T)}
        out['rna_seq'] = {128: torch.zeros(B, S128, T), 1: torch.zeros(B, dna.shape[1], T)}
        out['dnase'] = {128: torch.zeros(B, S128, T), 1: torch.zeros(B, dna.shape[1], T)}
        out['contact_maps'] = torch.zeros(B, S128, S128, T)
        return out


def _make_model() -> VariantScoringModel:
    model = _RecordingModel()
    # Skip device move; feed a tiny tensor through predict directly.
    sm = VariantScoringModel.__new__(VariantScoringModel)
    sm.model = model
    sm.num_organisms = 2
    sm.organism_map = {'human': 0, 'homo_sapiens': 0, 'mouse': 1, 'mus_musculus': 1}
    sm.default_organism_index = 0
    sm.device = torch.device('cpu')
    sm._fasta_extractor = None
    sm._gene_annotation = None
    sm._polya_annotation = None
    sm._track_metadata = {0: {}, 1: {}}
    return sm


@pytest.mark.unit
def test_required_heads_default_uses_requested_output():
    scorer = CenterMaskScorer(OutputType.ATAC, 501, AggregationType.DIFF_LOG2_SUM)
    assert scorer.required_heads == frozenset({'atac'})

    rna = GeneMaskLFCScorer(OutputType.RNA_SEQ)
    assert rna.required_heads == frozenset({'rna_seq'})

    contact = ContactMapScorer()
    assert contact.required_heads == frozenset({'contact_maps'})


@pytest.mark.unit
def test_required_heads_splice_junction_only_splice_sites():
    # The junction head is recomputed in the unified-splicing second pass, so
    # the first pass only needs splice_sites to derive unified positions.
    sj = SpliceJunctionScorer()
    assert sj.required_heads == frozenset({'splice_sites'})


@pytest.mark.unit
def test_required_heads_splicing_scorer_uses_requested_output():
    s1 = GeneMaskSplicingScorer(OutputType.SPLICE_SITES, width=None)
    assert s1.required_heads == frozenset({'splice_sites'})
    s2 = GeneMaskSplicingScorer(OutputType.SPLICE_SITE_USAGE, width=None)
    assert s2.required_heads == frozenset({'splice_site_usage'})


@pytest.mark.unit
def test_score_variant_forwards_union_of_required_heads(monkeypatch):
    """When score_variant runs, model.forward must receive heads union."""
    sm = _make_model()

    # Bypass the FASTA-driven path: monkeypatch predict_variant to call predict()
    # with the constructed heads kwarg, exactly as the real flow does, but with
    # a synthetic sequence so we don't need a fasta.
    seq_len = 256

    def fake_predict_variant(interval, variant, organism=None, to_cpu=False,
                             unified_splicing=False, heads=None):
        kwargs = {'return_embeddings': unified_splicing}
        if heads is not None:
            kwargs['heads'] = heads
        seq = torch.zeros(1, seq_len, 4)
        ref = sm.model(seq, torch.zeros(1, dtype=torch.long), **kwargs)
        alt = sm.model(seq, torch.zeros(1, dtype=torch.long), **kwargs)
        return ref, alt

    sm.predict_variant = fake_predict_variant

    interval = Interval('chr1', 0, seq_len)
    variant = Variant('chr1', 100, 'A', 'C')

    scorers = [
        CenterMaskScorer(OutputType.ATAC, 501, AggregationType.DIFF_LOG2_SUM),
        GeneMaskActiveScorer(OutputType.RNA_SEQ),  # included so required_heads adds rna_seq; its score() still runs, but the stub annotation below returns no genes so it finishes quickly
    ]

    # GeneMaskActiveScorer requires gene_annotation; supply a stub to satisfy it
    class _StubAnno:
        def get_genes_in_interval(self, interval, gene_types=None):
            return []

    sm._gene_annotation = _StubAnno()

    sm.score_variant(interval, variant, scorers)

    # The model should have been called with heads being the union {atac, rna_seq}
    assert sm.model.last_heads_seen is not None
    assert set(sm.model.last_heads_seen) == {'atac', 'rna_seq'}


@pytest.mark.unit
def test_score_variant_subset_only_atac():
    sm = _make_model()
    seq_len = 256

    def fake_predict_variant(interval, variant, organism=None, to_cpu=False,
                             unified_splicing=False, heads=None):
        kwargs = {'return_embeddings': unified_splicing}
        if heads is not None:
            kwargs['heads'] = heads
        seq = torch.zeros(1, seq_len, 4)
        ref = sm.model(seq, torch.zeros(1, dtype=torch.long), **kwargs)
        alt = sm.model(seq, torch.zeros(1, dtype=torch.long), **kwargs)
        return ref, alt

    sm.predict_variant = fake_predict_variant

    scorers = [CenterMaskScorer(OutputType.ATAC, 501, AggregationType.DIFF_LOG2_SUM)]
    sm.score_variant(Interval('chr1', 0, seq_len), Variant('chr1', 100, 'A', 'C'), scorers)
    assert set(sm.model.last_heads_seen) == {'atac'}


@pytest.mark.unit
def test_score_variant_empty_scorers_runs_all_heads():
    """If no scorers declare requirements, heads is None (run everything)."""
    sm = _make_model()
    seq_len = 256

    captured = {}

    def fake_predict_variant(interval, variant, organism=None, to_cpu=False,
                             unified_splicing=False, heads=None):
        captured['heads'] = heads
        # Return minimal outputs (won't be consumed since scorers list is empty)
        return {}, {}

    sm.predict_variant = fake_predict_variant
    sm.score_variant(Interval('chr1', 0, seq_len), Variant('chr1', 100, 'A', 'C'), [])
    assert captured['heads'] is None

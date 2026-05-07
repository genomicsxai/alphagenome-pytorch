"""Tests for the unified ISM SNV-generation helpers used by both the local
inference path and the serving extension's `VariantScorer.score_ism_variants`.
"""

import pytest

from alphagenome_pytorch.variant_scoring.inference import (
    _build_ism_variants,
    _resolve_ism_interval,
)
from alphagenome_pytorch.variant_scoring.types import Interval, Variant


# ---------------------------------------------------------------------------
# _build_ism_variants
# ---------------------------------------------------------------------------


def test_build_ism_variants_three_alts_per_position():
    interval = Interval('chr1', 100, 110)
    sequence = 'ACGTACGTAC'  # positions 100..109
    ism_interval = Interval('chr1', 102, 105)  # positions 102 (G), 103 (T), 104 (A)

    variants = _build_ism_variants(sequence, interval, ism_interval)

    assert len(variants) == 9
    # Variants are ordered by genomic position, then by alt-base order in 'ACGT'.
    expected = [
        # pos 102 (1-based 103), REF='G' -> A,C,T
        Variant('chr1', 103, 'G', 'A'),
        Variant('chr1', 103, 'G', 'C'),
        Variant('chr1', 103, 'G', 'T'),
        # pos 103 (1-based 104), REF='T' -> A,C,G
        Variant('chr1', 104, 'T', 'A'),
        Variant('chr1', 104, 'T', 'C'),
        Variant('chr1', 104, 'T', 'G'),
        # pos 104 (1-based 105), REF='A' -> C,G,T
        Variant('chr1', 105, 'A', 'C'),
        Variant('chr1', 105, 'A', 'G'),
        Variant('chr1', 105, 'A', 'T'),
    ]
    assert variants == expected


def test_build_ism_variants_skips_non_acgt_reference():
    interval = Interval('chr1', 100, 105)
    sequence = 'ANCGT'  # position 101 has 'N'
    ism_interval = Interval('chr1', 100, 103)

    variants = _build_ism_variants(sequence, interval, ism_interval)

    # Position 101 (the 'N') skipped entirely.
    positions = sorted({v.position for v in variants})
    assert positions == [101, 103]  # 1-based: 100->101 (A), skip 101->102 (N), 102->103 (C)


def test_build_ism_variants_uses_custom_variant_cls():
    """The helper must instantiate via the supplied factory.

    Serving passes ``genome.Variant`` so the result is the official type
    without requiring a back-conversion from PT ``Variant``.
    """
    captured: list[dict] = []

    class StubVariant:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    interval = Interval('chr1', 100, 102)
    ism_interval = Interval('chr1', 100, 102)
    sequence = 'AT'

    out = _build_ism_variants(
        sequence, interval, ism_interval,
        variant_cls=StubVariant,
    )

    assert all(isinstance(v, StubVariant) for v in out)
    assert len(captured) == 6  # 2 positions x 3 alts
    assert captured[0] == {
        'chromosome': 'chr1',
        'position': 101,  # 1-based
        'reference_bases': 'A',
        'alternate_bases': 'C',
    }


def test_build_ism_variants_respects_nucleotides_subset():
    interval = Interval('chr1', 100, 102)
    sequence = 'AT'
    ism_interval = Interval('chr1', 100, 102)

    # Restrict to purines: A->G only at position 0; T skipped (not in nucleotides).
    variants = _build_ism_variants(sequence, interval, ism_interval, nucleotides='AG')

    assert variants == [Variant('chr1', 101, 'A', 'G')]


# ---------------------------------------------------------------------------
# _resolve_ism_interval
# ---------------------------------------------------------------------------


def test_resolve_with_explicit_ism_interval_returns_it():
    interval = Interval('chr1', 0, 1000)
    ism_interval = Interval('chr1', 500, 510)

    out = _resolve_ism_interval(
        interval=interval, ism_interval=ism_interval,
        center_position=None, window_size=21,
    )
    assert out == ism_interval


def test_resolve_with_center_position_constructs_centered_window():
    interval = Interval('chr1', 0, 1000)

    # 1-based center 501, window_size 21 -> 0-based [500-10, 500-10+21) = [490, 511)
    out = _resolve_ism_interval(
        interval=interval, ism_interval=None,
        center_position=501, window_size=21,
    )
    assert out.chromosome == 'chr1'
    assert out.start == 490
    assert out.end == 511
    assert out.end - out.start == 21


def test_resolve_rejects_both_modes():
    interval = Interval('chr1', 0, 1000)
    ism_interval = Interval('chr1', 500, 510)
    with pytest.raises(ValueError, match='only one of'):
        _resolve_ism_interval(
            interval=interval, ism_interval=ism_interval,
            center_position=505, window_size=21,
        )


def test_resolve_rejects_neither_mode():
    interval = Interval('chr1', 0, 1000)
    with pytest.raises(ValueError, match='either ism_interval or center_position'):
        _resolve_ism_interval(
            interval=interval, ism_interval=None,
            center_position=None, window_size=21,
        )


def test_resolve_rejects_chromosome_mismatch():
    interval = Interval('chr1', 0, 1000)
    ism_interval = Interval('chr2', 100, 110)
    with pytest.raises(ValueError, match='chromosome'):
        _resolve_ism_interval(
            interval=interval, ism_interval=ism_interval,
            center_position=None, window_size=21,
        )


def test_resolve_rejects_out_of_bounds():
    interval = Interval('chr1', 100, 200)
    too_low = Interval('chr1', 50, 110)
    too_high = Interval('chr1', 190, 250)
    with pytest.raises(ValueError, match='contained within'):
        _resolve_ism_interval(
            interval=interval, ism_interval=too_low,
            center_position=None, window_size=21,
        )
    with pytest.raises(ValueError, match='contained within'):
        _resolve_ism_interval(
            interval=interval, ism_interval=too_high,
            center_position=None, window_size=21,
        )


def test_resolve_rejects_negative_strand_ism_interval():
    interval = Interval('chr1', 0, 1000)
    neg = Interval('chr1', 500, 510, strand='-')
    with pytest.raises(ValueError, match='positive strand'):
        _resolve_ism_interval(
            interval=interval, ism_interval=neg,
            center_position=None, window_size=21,
        )


# ---------------------------------------------------------------------------
# Equivalence: center+window sugar matches explicit ism_interval
# ---------------------------------------------------------------------------


def test_center_window_and_ism_interval_produce_equivalent_variants():
    """The center+window sugar is just a way to construct an ism_interval —
    given the equivalent interval, the SNV lists must match exactly.
    """
    interval = Interval('chr1', 0, 1000)
    sequence = 'ACGT' * 250  # length 1000

    # 1-based center 501, window 21 -> 0-based [490, 511)
    via_center = _resolve_ism_interval(
        interval=interval, ism_interval=None,
        center_position=501, window_size=21,
    )
    via_explicit = Interval('chr1', 490, 511)

    v1 = _build_ism_variants(sequence, interval, via_center)
    v2 = _build_ism_variants(sequence, interval, via_explicit)

    assert v1 == v2
    # Sanity: 21 positions x 3 alts = 63 variants.
    assert len(v1) == 21 * 3

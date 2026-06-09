"""Math parity tests for `ism_matrix` against upstream `alphagenome.interpretation.ism.ism_matrix`.

The upstream contract: per row, subtract `sum(scores) / (V - 1)`. With REF column = 0
and three ALT columns filled, this equals the mean of the three ALT scores. After
mean-centering, REF holds `-mean(ALT)` and each ALT holds `score - mean(ALT)`. With
`multiply_by_sequence=True`, only the REF column survives.

Coverage guard: a real ACGT reference contributes exactly 3 ALT bases, so a scored
row holds 3 (normal) or 4 (an `N` reference yields all four ACGT as alts). A row with
only 1 or 2 scored ALTs would be mis-centered (incomplete sum still divided by V-1),
so `_ism_matrix` raises `ValueError`. Rows with 0 scored ALTs (window edges) are left
as zeros.
"""

import numpy as np
import pytest
import torch

from alphagenome_pytorch.variant_scoring.inference import _ism_matrix
from alphagenome_pytorch.variant_scoring.types import Interval, Variant


def _build_snv_inputs():
    """Two-position interval with hand-picked SNV scores.

    Position 100 (1-based 101), REF='A':  A->C=2.0, A->G=4.0, A->T=6.0
    Position 101 (1-based 102), REF='T':  T->A=1.0, T->C=3.0, T->G=5.0
    """
    interval = Interval(chromosome='chr1', start=100, end=102)
    variants = [
        Variant('chr1', 101, 'A', 'C'), Variant('chr1', 101, 'A', 'G'), Variant('chr1', 101, 'A', 'T'),
        Variant('chr1', 102, 'T', 'A'), Variant('chr1', 102, 'T', 'C'), Variant('chr1', 102, 'T', 'G'),
    ]
    scores = [2.0, 4.0, 6.0, 1.0, 3.0, 5.0]
    return interval, variants, scores


def test_pre_mask_matches_upstream_centering():
    """Without multiply_by_sequence: matrix = score - sum(row)/(V-1) per cell.

    Upstream behavior: REF column starts at 0, so REF cell becomes `-mean(ALT scores)`
    and each ALT cell becomes `score - mean(ALT scores)`.
    """
    interval, variants, scores = _build_snv_inputs()

    matrix = _ism_matrix(scores, variants, interval, multiply_by_sequence=False)
    matrix = matrix.numpy() if isinstance(matrix, torch.Tensor) else matrix

    # Position 100 (REF='A'): sum = 0+2+4+6 = 12, mean = 4
    # vocab order ACGT -> [A=-4, C=-2, G=0, T=2]
    assert np.allclose(matrix[0], [-4.0, -2.0, 0.0, 2.0])

    # Position 101 (REF='T'): sum = 1+3+5+0 = 9, mean = 3
    # vocab order ACGT -> [A=-2, C=0, G=2, T=-3]
    assert np.allclose(matrix[1], [-2.0, 0.0, 2.0, -3.0])


def test_post_mask_keeps_only_reference_column():
    """multiply_by_sequence=True (default) zeros ALT columns, keeps REF."""
    interval, variants, scores = _build_snv_inputs()

    matrix = _ism_matrix(scores, variants, interval, multiply_by_sequence=True)
    matrix = matrix.numpy() if isinstance(matrix, torch.Tensor) else matrix

    # Position 100 (REF='A'): only A column non-zero, value = -mean(ALT) = -4
    assert np.allclose(matrix[0], [-4.0, 0.0, 0.0, 0.0])

    # Position 101 (REF='T'): only T column non-zero, value = -mean(ALT) = -3
    assert np.allclose(matrix[1], [0.0, 0.0, 0.0, -3.0])


def test_regression_centering_is_sum_over_three_not_nine():
    """Anti-regression for the prior 3x bug.

    The previous implementation subtracted `mean(filled)/3` = `sum/9` instead of
    `sum/3`, giving values 1/3 the upstream magnitude. Lock in the upstream value.
    """
    interval, variants, scores = _build_snv_inputs()

    matrix = _ism_matrix(scores, variants, interval, multiply_by_sequence=True)
    matrix = matrix.numpy() if isinstance(matrix, torch.Tensor) else matrix

    # Position 100, REF='A' column: must be -sum/3 = -12/3 = -4.0,
    # NOT the buggy -sum/9 = -12/9 ≈ -1.333.
    assert matrix[0, 0] == pytest.approx(-4.0)
    assert matrix[0, 0] != pytest.approx(-12.0 / 9)

    # Position 101, REF='T' column: -sum/3 = -9/3 = -3.0, NOT -9/9 = -1.0.
    assert matrix[1, 3] == pytest.approx(-3.0)
    assert matrix[1, 3] != pytest.approx(-1.0)


def test_matches_upstream_vectorized_formula():
    """Direct equality with the upstream `np.sum(scores, axis=-1, keepdims=True) / 3` form."""
    interval, variants, scores = _build_snv_inputs()
    vocabulary = 'ACGT'

    # Reproduce upstream construction inline.
    expected = np.zeros((interval.width, len(vocabulary)), dtype=np.float32)
    filled = np.zeros((interval.width, len(vocabulary)), dtype=bool)
    base_index = {b: i for i, b in enumerate(vocabulary)}
    for v, s in zip(variants, scores):
        pos = v.start - interval.start
        expected[pos, base_index[v.alternate_bases]] = s
        filled[pos, base_index[v.alternate_bases]] = True
    expected -= expected.sum(axis=-1, keepdims=True) / (len(vocabulary) - 1)
    expected_masked = expected * (~filled).astype(np.float32)

    got_pre = _ism_matrix(scores, variants, interval, multiply_by_sequence=False).numpy()
    got_post = _ism_matrix(scores, variants, interval, multiply_by_sequence=True).numpy()

    np.testing.assert_allclose(got_pre, expected, atol=1e-6)
    np.testing.assert_allclose(got_post, expected_masked, atol=1e-6)


def test_partially_filled_position_raises():
    """A position with only 1 or 2 scored ALT bases is rejected.

    Centering divides by (V-1)=3 regardless of how many cells are filled, so a
    partially scored position would silently produce a wrong-magnitude row. The
    coverage guard raises instead, reporting the offending 1-based position.
    """
    interval = Interval(chromosome='chr1', start=100, end=101)
    variants = [Variant('chr1', 101, 'A', 'C'), Variant('chr1', 101, 'A', 'G')]  # missing A->T
    scores = [3.0, 6.0]

    with pytest.raises(ValueError, match=r'1 or 2 alternate bases.*101'):
        _ism_matrix(scores, variants, interval, multiply_by_sequence=False)


def test_n_reference_four_alts_allowed():
    """A non-ACGT (N) reference yields all 4 ALT bases and is allowed.

    A real ACGT base can only have 3 alternates, so 4 scored ALTs uniquely signals
    an N reference. It must not trip the partial-coverage guard. Under the default
    `multiply_by_sequence=True` mask, every column is filled so the row zeros out.
    """
    interval = Interval(chromosome='chr1', start=100, end=101)
    variants = [
        Variant('chr1', 101, 'N', 'A'), Variant('chr1', 101, 'N', 'C'),
        Variant('chr1', 101, 'N', 'G'), Variant('chr1', 101, 'N', 'T'),
    ]
    scores = [1.0, 2.0, 3.0, 4.0]

    # Default mask: all four columns filled -> whole row zeroed, no error.
    post = _ism_matrix(scores, variants, interval, multiply_by_sequence=True).numpy()
    assert np.allclose(post[0], [0.0, 0.0, 0.0, 0.0])

    # Pre-mask: centered by sum/(V-1)=10/3 per cell, still no error.
    pre = _ism_matrix(scores, variants, interval, multiply_by_sequence=False).numpy()
    assert np.allclose(pre[0], np.array([1.0, 2.0, 3.0, 4.0]) - 10.0 / 3)


def test_skips_non_snv_and_oob_variants():
    """Indels and out-of-interval variants are silently ignored.

    Position 0 still gets its full 3 SNVs (so the coverage guard is satisfied); the
    insertion and the out-of-interval SNV must not contribute. Position 1 stays empty
    (0 filled is allowed — it is left as zeros, not raised).
    """
    interval = Interval(chromosome='chr1', start=100, end=102)
    variants = [
        Variant('chr1', 101, 'A', 'C'),   # in-interval SNV
        Variant('chr1', 101, 'A', 'G'),   # in-interval SNV
        Variant('chr1', 101, 'A', 'T'),   # in-interval SNV
        Variant('chr1', 101, 'A', 'CG'),  # insertion: skipped
        Variant('chr1', 200, 'A', 'C'),   # out of interval: skipped
    ]
    scores = [2.0, 4.0, 6.0, 99.0, 99.0]

    matrix = _ism_matrix(scores, variants, interval, multiply_by_sequence=False).numpy()

    # Only the 3 valid SNVs counted: position 0, sum = 12, subtract 12/3 = 4 per cell.
    # ACGT -> [A=-4, C=-2, G=0, T=2]; the skipped variants leave no trace.
    assert np.allclose(matrix[0], [-4.0, -2.0, 0.0, 2.0])
    # Position 1 untouched: all zero.
    assert np.allclose(matrix[1], [0.0, 0.0, 0.0, 0.0])

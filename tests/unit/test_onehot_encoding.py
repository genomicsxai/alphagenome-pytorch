"""Tests for DNA sequence ↔ one-hot encoding roundtrips and edge cases."""

import numpy as np
import pytest

from alphagenome_pytorch.utils.sequence import (
    onehot_to_sequence,
    sequence_to_onehot,
)


@pytest.mark.parametrize("seq", ["A", "ACGT", "AAAA", "TTTT", "ACGTACGTACGTACGT"])
def test_roundtrip(seq):
    assert onehot_to_sequence(sequence_to_onehot(seq)) == seq


def test_lowercase_roundtrip():
    """Lowercase input should encode identically to uppercase."""
    assert onehot_to_sequence(sequence_to_onehot("acgt")) == "ACGT"


def test_n_encodes_as_zeros_and_decodes_back():
    oh = sequence_to_onehot("ANGT")
    np.testing.assert_array_equal(oh[0], [1, 0, 0, 0])
    np.testing.assert_array_equal(oh[1], [0, 0, 0, 0])
    np.testing.assert_array_equal(oh[2], [0, 0, 1, 0])
    np.testing.assert_array_equal(oh[3], [0, 0, 0, 1])
    assert onehot_to_sequence(oh) == "ANGT"


def test_all_n():
    oh = sequence_to_onehot("NNN")
    assert oh.sum() == 0
    assert onehot_to_sequence(oh) == "NNN"


def test_empty_string():
    oh = sequence_to_onehot("")
    assert oh.shape == (0, 4)
    assert onehot_to_sequence(oh) == ""


def test_dtype_and_shape():
    oh = sequence_to_onehot("ACGT")
    assert oh.dtype == np.uint8
    assert oh.shape == (4, 4)


def test_each_row_sums_to_one_or_zero():
    oh = sequence_to_onehot("ACNGTNA")
    row_sums = oh.sum(axis=1)
    assert all(s in (0, 1) for s in row_sums)


def test_onehot_to_sequence_rejects_wrong_shape():
    with pytest.raises(ValueError, match="Expected shape"):
        onehot_to_sequence(np.zeros((4, 3), dtype=np.uint8))

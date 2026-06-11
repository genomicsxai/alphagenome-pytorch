from __future__ import annotations

import numpy as np

from alphagenome_pytorch.genome import (
    GenomeSequenceSource,
    Interval,
    Variant,
    apply_variant_to_sequence,
)


def _write_fasta(path):
    path.write_text(">chr1\nACGTNN\n>2\nTTAA\n")


def test_genome_sequence_source_fetches_sequences_and_chrom_sizes(tmp_path):
    fasta_path = tmp_path / "genome.fa"
    _write_fasta(fasta_path)

    source = GenomeSequenceSource(fasta_path)

    assert source.chrom_sizes["chr1"] == 6
    assert source.fetch_sequence(Interval("chr1", 0, 4)) == "ACGT"
    assert source.fetch_sequence(Interval("chr2", 1, 3)) == "TA"


def test_genome_sequence_source_reopens_stale_fasta_handle(tmp_path):
    fasta_path = tmp_path / "genome.fa"
    _write_fasta(fasta_path)
    source = GenomeSequenceSource(fasta_path)

    first = source.fasta
    source._owner_pid = -1
    second = source.fasta

    assert second is not first
    assert source.fetch_sequence(Interval("chr1", 0, 1)) == "A"


def test_genome_sequence_source_cached_fetch_and_uniform_padding(tmp_path):
    fasta_path = tmp_path / "genome.fa"
    _write_fasta(fasta_path)

    source = GenomeSequenceSource(
        fasta_path,
        chromosomes={"chr1"},
        cache=True,
        ambiguous="uniform",
    )

    values = source.fetch_onehot("chr1", -2, 3, pad=True, ambiguous="uniform")
    assert values.shape == (5, 4)
    assert np.allclose(values[0], [0.25, 0.25, 0.25, 0.25])
    assert np.allclose(values[1], [0.25, 0.25, 0.25, 0.25])
    assert np.allclose(values[2], [1, 0, 0, 0])
    assert np.allclose(values[3], [0, 1, 0, 0])
    assert np.allclose(values[4], [0, 0, 1, 0])


def test_apply_variant_to_sequence_checks_reference():
    interval = Interval("chr1", 0, 6)
    variant = Variant("chr1", 2, "C", "T")

    assert apply_variant_to_sequence("ACGTNN", variant, interval) == "ATGTNN"

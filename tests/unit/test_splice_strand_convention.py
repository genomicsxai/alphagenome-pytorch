"""Splice-site roles follow AlphaGenome's strand convention (minus-strand donor = right site)."""


import pandas as pd
import pytest

from alphagenome_pytorch.extensions.finetuning import star_junctions as sj

# One + and one - junction, both with intron 101..200 (exon_start=100, exon_end=201).
JUNCS = pd.DataFrame({
    "chrom": ["chr1", "chr1"],
    "exon_start": [100, 100],
    "exon_end": [201, 201],
    "strand": ["+", "-"],
    "count": [3.0, 5.0],
})


def test_donor_acceptor_positions_match_alphagenome():
    donor, acceptor = sj.donor_acceptor_positions(JUNCS)
    assert donor.tolist() == [100, 201]
    assert acceptor.tolist() == [201, 100]
    genome = pytest.importorskip("alphagenome.data.genome")
    for strand, d, a in (("+", 100, 201), ("-", 201, 100)):
        # alphagenome Junction: 0-based [start, end) intron, donor/acceptor at its ends.
        j = genome.Junction("chr1", 100, 200, strand)
        # Exonic 1-based sites flank the intron: start side -> exon_start, end side -> exon_end.
        side = {j.start: 100, j.end: 201}
        assert side[j.donor] == d and side[j.acceptor] == a


def test_classification_array_minus_strand():
    arr = sj.junctions_to_classification_array([JUNCS], "chr1", 0, 300)
    # 0-based index = 1-based position - 1
    assert arr[99].argmax() == 0 and arr[200].argmax() == 1       # + : donor left, acceptor right
    neg = sj.junctions_to_classification_array([JUNCS.iloc[[1]]], "chr1", 0, 300)
    assert neg[200].argmax() == 2 and neg[99].argmax() == 3       # - : donor right, acceptor left


def test_junction_matrix_minus_strand():
    cls = sj.junctions_to_classification_array([JUNCS], "chr1", 0, 300)
    pos, mat = sj.junctions_to_junction_matrix([JUNCS], max_splice_sites=4, cls_arr=cls,
                                               chrom="chr1", start=0, seq_len=300)
    assert pos[:, 0].tolist() == [99, 200, 200, 99]   # [+donor, +acceptor, -donor, -acceptor]
    assert mat[0, 0, 0] == 3.0 and mat[0, 0, 1] == 5.0


def test_site_table_and_alpha_beta2_roles():
    sites = sj.junctions_to_site_table(JUNCS).set_index(["strand", "role"])["position"]
    assert sites[("-", "donor")] == 201 and sites[("-", "acceptor")] == 100
    donor_alpha, acceptor_alpha, _, _ = sj.compute_alpha_beta2(JUNCS)
    assert donor_alpha[("chr1", 201, "-")] == 5.0
    assert acceptor_alpha[("chr1", 100, "-")] == 5.0


def test_filter_intervals_is_inclusive_of_window_end():
    j = pd.DataFrame({"chrom": ["chr1"], "exon_start": [1], "exon_end": [300], "strand": ["+"]})
    iv = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [300]})
    kept, _ = sj.filter_intervals_with_junctions(iv, j)
    assert len(kept) == 1                      # sites at 0-based 0 and 299 are inside [0, 300)
    kept, _ = sj.filter_intervals_with_junctions(iv.assign(start=1), j)
    assert len(kept) == 0                      # 1-based 1 is 0-based 0, outside [1, 300)


def test_site_cap_keeps_best_supported():
    juncs = pd.DataFrame({"chrom": "chr1", "exon_start": [10, 20, 30], "exon_end": [40, 50, 60],
                          "strand": "+", "count": [1.0, 9.0, 5.0]})
    cls = sj.junctions_to_classification_array([juncs], "chr1", 0, 100)
    pos, _ = sj.junctions_to_junction_matrix([juncs], max_splice_sites=2, cls_arr=cls,
                                             chrom="chr1", start=0, seq_len=100)
    assert pos[0].tolist() == [19, 29]        # donors with 9 and 5 reads, genomic order

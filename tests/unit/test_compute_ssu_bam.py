"""compute_ssu.py: acceptor alpha/beta2 bookkeeping on a tiny BAM."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pysam = pytest.importorskip("pysam")

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compute_ssu.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("compute_ssu", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_bam(path, reads):
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 1000}]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for name, start, cigar in reads:
            a = pysam.AlignedSegment()
            a.query_name = name
            a.query_sequence = "A" * 50
            a.flag = 16                      # unpaired, reverse -> "+" for strandedType "rf"
            a.reference_id = 0
            a.reference_start = start
            a.mapping_quality = 60
            a.cigarstring = cigar
            a.query_qualities = pysam.qualitystring_to_array("I" * 50)
            out.write(a)
    pysam.index(str(path))


def test_competing_acceptor_one_bp_away_is_beta2_not_alpha(tmp_path):
    mod = _load_script()
    # Two + strand junctions sharing donor 100: introns [100,200) and [100,201)
    # (0-based half-open), i.e. acceptors at iv_e = 200 and 201.
    junctions = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "strand": ["+", "+"],
        "exon_start": [100, 100],
        "exon_end": [201, 202],
        "count": [1, 1],
    })
    bam = tmp_path / "reads.bam"
    _write_bam(bam, [
        ("uses_200", 75, "25M100N25M"),      # intron [100, 200)
        ("uses_201", 75, "25M101N25M"),      # intron [100, 201): competitor of acceptor 200
    ])
    out = mod.compute_spliser_counts(bam, junctions, mapq_min=30, strandedType="rf")
    # Output positions are 1-based exonic: acceptors at iv_e + 1 = 201 and 202.
    acc = out[(out["role"] == "acceptor") & (out["strand"] == "+")].set_index("position")
    assert acc.loc[201, "alpha_bam"] == 1
    assert acc.loc[201, "beta2_bam"] == 1      # the 1-bp-shifted acceptor read competes
    assert acc.loc[202, "alpha_bam"] == 1
    assert acc.loc[202, "beta2_bam"] == 1

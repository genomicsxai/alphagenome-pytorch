"""SplicingDataset works inside MultimodalDataset (augmentation interface)."""

import numpy as np
import pytest

pytest.importorskip("pyfaidx")

from alphagenome_pytorch.extensions.finetuning.datasets import (
    MultimodalDataset,
    SplicingDataset,
)


@pytest.fixture
def splice_files(tmp_path):
    rng = np.random.default_rng(0)
    fasta = tmp_path / "g.fa"
    fasta.write_text(">chr1\n" + "".join(rng.choice(list("ACGT"), 2000)) + "\n")
    sj = tmp_path / "s1.SJ.out.tab"
    # chrom, intron_start, intron_end, strand(1=+,2=-), motif, annotated, unique, multi, overhang
    sj.write_text("chr1\t900\t1000\t1\t1\t1\t5\t0\t20\nchr1\t1050\t1100\t2\t1\t1\t3\t0\t20\n")
    bed = tmp_path / "w.bed"
    bed.write_text("chr1\t900\t1100\nchr1\t1\t3\n")   # 2nd window runs off chr1 start
    return str(fasta), str(sj), str(bed)


def test_splicing_dataset_in_multimodal(splice_files):
    fasta, sj, bed = splice_files
    ds = SplicingDataset(fasta, bed, [sj], sequence_length=512, filter_to_junctions=False)
    assert len(ds) == 1                      # off-chromosome window skipped
    assert not ds.augments
    multi = MultimodalDataset({"splice_site": ds, "splice_junctions": ds})
    seq, targets = multi[0]
    assert seq.shape == (512, 4)
    assert set(targets) == {"splice_site", "splice_junctions"}
    assert "junction_matrix" in targets["splice_junctions"]
    with pytest.raises(NotImplementedError):
        ds.get(0, shift=3)

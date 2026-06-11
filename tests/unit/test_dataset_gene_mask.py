"""Tests for gene_mask integration in GenomicDataset / MultimodalDataset.

Verifies that:
  - Without `gene_mask_extractor`, datasets keep their original 2-tuple
    return shape (no behavioral regression).
  - With an extractor + g_max, GenomicDataset returns a 3-tuple whose
    third element is a `[S, 2, g_max]` bool tensor padded along the
    gene axis.
  - MultimodalDataset propagates gene_mask from the first dataset that
    yields one (gene_mask is sample-level, not per-modality).
  - g_max overflow raises a clear error.
  - Forgetting g_max when extractor is set raises in the constructor.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pyBigWig = pytest.importorskip("pyBigWig")

from alphagenome_pytorch.extensions.finetuning.datasets import (
    GenomicDataset,
    MultimodalDataset,
)
from alphagenome_pytorch.extensions.finetuning.gene_annotation import (
    GeneMaskExtractor,
)


CHROM = "chr1"
CHROM_SIZE = 8_192


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def fasta_file(tmp_dir):
    """Tiny FASTA with one chromosome of A's."""
    path = tmp_dir / "genome.fa"
    with open(path, "w") as f:
        f.write(f">{CHROM}\n")
        f.write("A" * CHROM_SIZE + "\n")
    # pyfaidx will create the .fai on first open; nothing else needed here.
    return str(path)


@pytest.fixture
def bigwig_file(tmp_dir):
    """Tiny bigwig with constant signal."""
    path = tmp_dir / "track.bw"
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([(CHROM, CHROM_SIZE)])
    bw.addEntries(
        [CHROM] * 4,
        [1000, 2000, 3000, 4000],
        ends=[1100, 2100, 3100, 4100],
        values=[1.0, 2.0, 3.0, 4.0],
    )
    bw.close()
    return str(path)


@pytest.fixture
def bed_file(tmp_dir):
    """One window centered at 4096 with width 2048."""
    path = tmp_dir / "regions.bed"
    with open(path, "w") as f:
        f.write(f"{CHROM}\t3072\t5120\n")
    return str(path)


@pytest.fixture
def extractor():
    """Two genes, one in the test window, one outside."""
    table = pd.DataFrame(
        {
            "Chromosome": [CHROM, CHROM],
            "Start": [3500, 7000],
            "End": [4500, 7500],
            "Strand": ["+", "-"],
            "gene_id": ["GENE-IN", "GENE-OUT"],
            "gene_name": ["in", "out"],
            "gene_type": ["protein_coding", "protein_coding"],
        }
    )
    return GeneMaskExtractor(table)


@pytest.mark.unit
class TestGenomicDatasetGeneMask:

    def test_no_extractor_returns_two_tuple(
        self, fasta_file, bigwig_file, bed_file
    ):
        ds = GenomicDataset(
            genome_fasta=fasta_file,
            bigwig_files=[bigwig_file],
            bed_file=bed_file,
            sequence_length=2048,
            resolutions=(1,),
        )
        result = ds[0]
        assert len(result) == 2
        seq, targets = result
        assert seq.shape == (2048, 4)
        assert isinstance(targets, dict)

    def test_with_extractor_returns_three_tuple(
        self, fasta_file, bigwig_file, bed_file, extractor
    ):
        ds = GenomicDataset(
            genome_fasta=fasta_file,
            bigwig_files=[bigwig_file],
            bed_file=bed_file,
            sequence_length=2048,
            resolutions=(1,),
            gene_mask_extractor=extractor,
            g_max=4,
        )
        result = ds[0]
        assert len(result) == 3
        seq, targets, gene_mask = result
        assert seq.shape == (2048, 4)
        assert gene_mask.shape == (2048, 2, 4)
        assert gene_mask.dtype == torch.bool

        # Window is [3072, 5120). GENE-IN spans [3500, 4500), so positions
        # [428, 1428) relative to window. + strand → axis 1 = 0, gene 0.
        assert gene_mask[:428, 0, 0].sum() == 0
        assert gene_mask[428:1428, 0, 0].all()
        assert gene_mask[1428:, 0, 0].sum() == 0
        # Other gene slots padded zero.
        assert gene_mask[:, :, 1:].sum() == 0

    def test_extractor_without_g_max_raises(
        self, fasta_file, bigwig_file, bed_file, extractor
    ):
        with pytest.raises(ValueError, match="g_max is None"):
            GenomicDataset(
                genome_fasta=fasta_file,
                bigwig_files=[bigwig_file],
                bed_file=bed_file,
                sequence_length=2048,
                resolutions=(1,),
                gene_mask_extractor=extractor,
                g_max=None,
            )

    def test_g_max_overflow_raises_at_getitem(
        self, fasta_file, bigwig_file, bed_file
    ):
        # Build an extractor with 5 genes contained in the window, but
        # tell the dataset g_max=2. __getitem__ must raise.
        table = pd.DataFrame(
            {
                "Chromosome": [CHROM] * 5,
                "Start": list(range(3500, 4000, 100)),
                "End": list(range(3550, 4050, 100)),
                "Strand": ["+"] * 5,
                "gene_id": [f"G{i}" for i in range(5)],
                "gene_name": [f"g{i}" for i in range(5)],
                "gene_type": ["protein_coding"] * 5,
            }
        )
        extractor = GeneMaskExtractor(table)
        ds = GenomicDataset(
            genome_fasta=fasta_file,
            bigwig_files=[bigwig_file],
            bed_file=bed_file,
            sequence_length=2048,
            resolutions=(1,),
            gene_mask_extractor=extractor,
            g_max=2,
        )
        with pytest.raises(ValueError, match="exceeding g_max"):
            ds[0]


@pytest.mark.unit
class TestMultimodalDatasetGeneMask:

    def test_no_extractor_anywhere_returns_two_tuple(
        self, fasta_file, bigwig_file, bed_file
    ):
        ds = GenomicDataset(
            genome_fasta=fasta_file, bigwig_files=[bigwig_file],
            bed_file=bed_file, sequence_length=2048, resolutions=(1,),
        )
        multi = MultimodalDataset({"atac": ds})
        result = multi[0]
        assert len(result) == 2

    def test_extractor_on_one_dataset_propagates(
        self, fasta_file, bigwig_file, bed_file, extractor
    ):
        # atac: no extractor; rna_seq: has extractor.
        atac = GenomicDataset(
            genome_fasta=fasta_file, bigwig_files=[bigwig_file],
            bed_file=bed_file, sequence_length=2048, resolutions=(1,),
        )
        rna = GenomicDataset(
            genome_fasta=fasta_file, bigwig_files=[bigwig_file],
            bed_file=bed_file, sequence_length=2048, resolutions=(1,),
            gene_mask_extractor=extractor, g_max=4,
        )
        # Order matters: primary is the first dataset (atac here, with no
        # extractor). The propagation logic should still fall through to
        # rna_seq and pick up its gene_mask.
        multi = MultimodalDataset({"atac": atac, "rna_seq": rna})
        result = multi[0]
        assert len(result) == 3
        seq, modality_targets, gene_mask = result
        assert set(modality_targets.keys()) == {"atac", "rna_seq"}
        assert gene_mask.shape == (2048, 2, 4)

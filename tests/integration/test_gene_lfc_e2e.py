"""End-to-end smoke test for the gene LFC training path.

Not a model-correctness test. Verifies that the dataset → DataLoader →
compute_finetuning_loss path threads `gene_mask` through cleanly when
configured: shapes line up, default_collate handles the 3-tuple, and a
backward pass through the gene LFC term produces a finite gradient.

Synthetic everything: tiny FASTA, one bigwig per RNA-seq track, a 2-line
BED, and a hand-rolled GTF with one protein-coding gene fully contained
in the test window.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from alphagenome_pytorch.extensions.finetuning.datasets import GenomicDataset
from alphagenome_pytorch.extensions.finetuning.gene_annotation import (
    GeneMaskExtractor,
    derive_g_max,
    load_gene_table,
)
from alphagenome_pytorch.extensions.finetuning.training import (
    compute_finetuning_loss,
)
from alphagenome_pytorch.training import _build_strand_channel_mask

pyBigWig = pytest.importorskip("pyBigWig")
pyfaidx = pytest.importorskip("pyfaidx")
pyranges = pytest.importorskip("pyranges")


CHROM = "chr1"
CHROM_SIZE = 4096
SEQ_LEN = 2048


@pytest.fixture
def setup_files(tmp_path: Path):
    """Build a tiny self-contained training corpus on disk.

    Layout:
      - hg38_tiny.fa: one chromosome of A's
      - track_{plus,minus,plus2,minus2}.bw: 4 bigwigs with simple ramps
      - regions.bed: one window centered at chr1:1024 (covering [0, 2048))
      - genes.gtf: one + strand and one - strand gene fully inside the window
    """
    fasta_path = tmp_path / "hg38_tiny.fa"
    with open(fasta_path, "w") as f:
        f.write(f">{CHROM}\n{'A' * CHROM_SIZE}\n")

    bw_paths = []
    # Bigwigs require strictly increasing, non-overlapping intervals.
    # Fixed positions across tracks; per-track signal differs by track index.
    starts = [200, 300, 400, 500, 600, 1100, 1200, 1300]
    ends = [220, 320, 420, 520, 620, 1120, 1220, 1320]
    for i, name in enumerate(["plus", "minus", "plus2", "minus2"]):
        path = tmp_path / f"track_{name}.bw"
        bw = pyBigWig.open(str(path), "w")
        bw.addHeader([(CHROM, CHROM_SIZE)])
        bw.addEntries(
            [CHROM] * len(starts),
            starts,
            ends=ends,
            values=[float(j + 1 + i) for j in range(len(starts))],
        )
        bw.close()
        bw_paths.append(str(path))

    bed_path = tmp_path / "regions.bed"
    with open(bed_path, "w") as f:
        f.write(f"{CHROM}\t0\t{SEQ_LEN}\n")
        f.write(f"{CHROM}\t{CHROM_SIZE - SEQ_LEN}\t{CHROM_SIZE}\n")

    # Two genes inside [0, 2048) — one plus, one minus, both protein_coding.
    gtf_path = tmp_path / "genes.gtf"
    with open(gtf_path, "w") as f:
        f.write(
            'chr1\tHAVANA\tgene\t101\t800\t.\t+\t.\t'
            'gene_id "GENE-PLUS"; gene_name "PLUS"; gene_type "protein_coding";\n'
        )
        f.write(
            'chr1\tHAVANA\tgene\t1001\t1700\t.\t-\t.\t'
            'gene_id "GENE-MINUS"; gene_name "MINUS"; gene_type "protein_coding";\n'
        )

    return {
        "fasta": str(fasta_path),
        "bigwigs": bw_paths,
        "bed": str(bed_path),
        "gtf": str(gtf_path),
    }


@pytest.mark.integration
def test_dataloader_yields_three_tuple_with_gene_mask(setup_files):
    """DataLoader's default_collate should batch the 3-tuple element-wise,
    yielding (sequence, targets_dict, gene_mask) as batched tensors."""
    table = load_gene_table(setup_files["gtf"])
    extractor = GeneMaskExtractor(table)

    # Project BED windows to fixed length, matching GenomicDataset's behavior.
    intervals = [(CHROM, 0, SEQ_LEN), (CHROM, CHROM_SIZE - SEQ_LEN, CHROM_SIZE)]
    g_max = derive_g_max(extractor, intervals)
    assert g_max >= 1  # at least one gene fits in window 0

    ds = GenomicDataset(
        genome_fasta=setup_files["fasta"],
        bigwig_files=setup_files["bigwigs"],
        bed_file=setup_files["bed"],
        sequence_length=SEQ_LEN,
        resolutions=(1,),
        gene_mask_extractor=extractor,
        g_max=g_max,
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert len(batch) == 3
    sequences, targets_dict, gene_mask = batch
    assert sequences.shape == (2, SEQ_LEN, 4)
    assert 1 in targets_dict
    assert targets_dict[1].shape == (2, SEQ_LEN, 4)  # 4 tracks
    assert gene_mask.shape == (2, SEQ_LEN, 2, g_max)
    assert gene_mask.dtype == torch.bool


@pytest.mark.integration
def test_compute_finetuning_loss_with_gene_mask_backward(setup_files):
    """End-to-end: build dataset, fetch a batch, run compute_finetuning_loss
    with gene LFC enabled, do a backward, assert finite gradient.

    The "predictions" here are a small learnable parameter tensor — we don't
    need a real model to exercise the loss-side wiring.
    """
    table = load_gene_table(setup_files["gtf"])
    extractor = GeneMaskExtractor(table)
    intervals = [(CHROM, 0, SEQ_LEN)]
    g_max = derive_g_max(extractor, intervals)

    ds = GenomicDataset(
        genome_fasta=setup_files["fasta"],
        bigwig_files=setup_files["bigwigs"],
        bed_file=setup_files["bed"],
        sequence_length=SEQ_LEN,
        resolutions=(1,),
        gene_mask_extractor=extractor,
        g_max=g_max,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    sequences, targets_dict, gene_mask = next(iter(loader))

    # Stand-in predictions: same shape as targets, requires_grad for backward.
    targets_1bp = targets_dict[1]
    preds_1bp = torch.rand_like(targets_1bp).requires_grad_(True)
    predictions = {1: preds_1bp}
    targets = {1: targets_1bp}

    strand_mask = _build_strand_channel_mask("+-+-")  # 4 tracks
    loss, loss_dict = compute_finetuning_loss(
        predictions=predictions,
        targets=targets,
        resolution_weights={1: 1.0},
        positional_weight=1.0,
        device=torch.device("cpu"),
        gene_mask=gene_mask,
        gene_loss_weight=0.1,  # paper value
        gene_cross_track_weight=5.0,
        strand_channel_mask=strand_mask,
    )
    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert "loss_gene_lfc" in loss_dict
    assert "loss_gene_total_count" in loss_dict
    assert "loss_gene_positional" in loss_dict

    loss.backward()
    assert preds_1bp.grad is not None
    assert torch.all(torch.isfinite(preds_1bp.grad))


@pytest.mark.integration
def test_loss_off_when_weight_zero(setup_files):
    """gene_loss_weight=0.0 with gene_mask present must give the exact same
    loss as the no-gene-mask path. Pins the "default-off" contract."""
    table = load_gene_table(setup_files["gtf"])
    extractor = GeneMaskExtractor(table)
    intervals = [(CHROM, 0, SEQ_LEN)]
    g_max = derive_g_max(extractor, intervals)

    ds_with = GenomicDataset(
        genome_fasta=setup_files["fasta"],
        bigwig_files=setup_files["bigwigs"],
        bed_file=setup_files["bed"],
        sequence_length=SEQ_LEN,
        resolutions=(1,),
        gene_mask_extractor=extractor,
        g_max=g_max,
    )
    ds_without = GenomicDataset(
        genome_fasta=setup_files["fasta"],
        bigwig_files=setup_files["bigwigs"],
        bed_file=setup_files["bed"],
        sequence_length=SEQ_LEN,
        resolutions=(1,),
    )

    sequences_w, targets_w, gene_mask = ds_with[0]
    sequences_wo, targets_wo = ds_without[0]
    # Targets/sequences identical regardless of gene-mask plumbing.
    torch.testing.assert_close(sequences_w, sequences_wo)
    torch.testing.assert_close(targets_w[1], targets_wo[1])

    preds = {1: torch.rand_like(targets_w[1]).unsqueeze(0) + 0.5}
    targets = {1: targets_w[1].unsqueeze(0)}
    strand_mask = _build_strand_channel_mask("+-+-")

    loss_no_gene, _ = compute_finetuning_loss(
        predictions=preds, targets=targets,
        resolution_weights={1: 1.0}, positional_weight=1.0,
        device=torch.device("cpu"),
    )
    loss_zero_w, _ = compute_finetuning_loss(
        predictions=preds, targets=targets,
        resolution_weights={1: 1.0}, positional_weight=1.0,
        device=torch.device("cpu"),
        gene_mask=gene_mask.unsqueeze(0),
        gene_loss_weight=0.0,
        strand_channel_mask=strand_mask,
    )
    torch.testing.assert_close(loss_no_gene, loss_zero_w)

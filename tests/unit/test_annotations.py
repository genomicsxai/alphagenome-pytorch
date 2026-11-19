"""Tests for the shared GTF/parquet annotation reader."""

from __future__ import annotations

import gzip

import pandas as pd
import pytest

from alphagenome_pytorch.annotations import (
    GeneAnnotation,
    annotation_format,
    read_annotation,
)
from alphagenome_pytorch.variant_scoring import Interval

_GTF = (
    "##description: test\n"
    'chr1\tHAVANA\tgene\t101\t200\t.\t+\t.\tgene_id "ENSG1";\n'
    'chr1\tHAVANA\texon\t101\t150\t.\t+\t.\tgene_id "ENSG1";\n'
)


@pytest.mark.parametrize("name, expected", [
    ("gencode.v46.annotation.gtf", "gtf"),
    ("gencode.v46.annotation.gtf.gz", "gtf"),
    ("annotation.txt", "gtf"),
    ("annotation", "gtf"),
    ("annotation.parquet", "parquet"),
    ("annotation.PQ", "parquet"),
])
def test_annotation_format(name, expected):
    assert annotation_format(name) == expected


def test_gzipped_parquet_is_rejected():
    with pytest.raises(ValueError, match="compressed parquet"):
        annotation_format("annotation.parquet.gz")


@pytest.mark.parametrize("name", ["mini.gtf", "mini.gtf.gz"])
def test_reads_gtf(tmp_path, name):
    pytest.importorskip("pyranges")
    path = tmp_path / name
    with (gzip.open if name.endswith(".gz") else open)(path, "wt") as fh:
        fh.write(_GTF)

    df = read_annotation(path)
    # GTF is 1-based inclusive; the frame is 0-based half-open.
    assert df[["Feature", "Start", "End"]].values.tolist() == [
        ["gene", 100, 200], ["exon", 100, 150],
    ]


def test_parquet_columns_are_projected(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "a.parquet"
    pd.DataFrame({"gene_id": ["ENSG1"], "Start": [0], "extra": [1]}).to_parquet(path)

    df = read_annotation(path, columns=["gene_id", "Start", "missing"])
    assert list(df.columns) == ["gene_id", "Start"]


def test_gene_annotation_reads_gzipped_gtf(tmp_path):
    pytest.importorskip("pyranges")
    path = tmp_path / "mini.gtf.gz"
    with gzip.open(path, "wt") as fh:
        fh.write(_GTF)

    ann = GeneAnnotation(path)
    assert ann.get_genes_in_interval(Interval("chr1", 0, 300)) == ["ENSG1"]
    assert ann._get_exons_for_gene("ENSG1") == [(100, 150)]


def test_gene_annotation_rejects_gzipped_parquet_at_construction(tmp_path):
    with pytest.raises(ValueError, match="compressed parquet"):
        GeneAnnotation(tmp_path / "annotation.parquet.gz")

"""Guard the top-level ``alphagenome_pytorch`` public API.

Existing tests import from submodules (e.g. ``alphagenome_pytorch.aggregation``),
so a regression that drops re-exports from the package ``__init__`` — as happened
when ``aggregate_genes`` and friends disappeared — is not otherwise caught. These
tests import from the top level on purpose.
"""

from __future__ import annotations

import importlib

import alphagenome_pytorch


# Aggregation exports need only torch (pandas/anndata are imported lazily inside
# the functions), so they must always import from the top-level package.
AGGREGATION_EXPORTS = (
    "aggregate_intervals",
    "aggregate_genes",
    "gene_expression",
    "gene_expression_values",
    "combine_gene_expression",
    "normalize_expression",
    "gene_expression_correlations",
    "GeneCounts",
    "GeneCountAccumulator",
)


def test_aggregation_exports_importable_from_top_level():
    module = importlib.import_module("alphagenome_pytorch")
    for name in ("AlphaGenome", *AGGREGATION_EXPORTS):
        assert hasattr(module, name), f"alphagenome_pytorch.{name} is missing"


def test_all_declares_the_public_api():
    exported = set(alphagenome_pytorch.__all__)
    for name in ("AlphaGenome", *AGGREGATION_EXPORTS):
        assert name in exported, f"{name} missing from __all__"
    # Lazy fine-tuning helpers are declared even though importing them may pull
    # optional dependencies (guarded by __getattr__), so check __all__ only.
    for name in ("TransferConfig", "load_trunk", "prepare_for_transfer"):
        assert name in exported, f"{name} missing from __all__"

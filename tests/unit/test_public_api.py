"""Guard the top-level ``alphagenome_pytorch`` public API.

The package root exposes ``AlphaGenome`` and the fine-tuning helpers
(``TransferConfig``, ``load_trunk``, ``prepare_for_transfer``). Aggregation is
intentionally NOT re-exported at the root — it lives under
``alphagenome_pytorch.aggregation``.
"""

from __future__ import annotations

import importlib

import alphagenome_pytorch


FINETUNING_HELPERS = ("TransferConfig", "load_trunk", "prepare_for_transfer")


def test_alphagenome_importable_from_top_level():
    module = importlib.import_module("alphagenome_pytorch")
    assert hasattr(module, "AlphaGenome")


def test_all_declares_the_public_api():
    exported = set(alphagenome_pytorch.__all__)
    assert "AlphaGenome" in exported
    # Lazy fine-tuning helpers are declared even though importing them may pull
    # optional dependencies (guarded by __getattr__), so check __all__ only.
    for name in FINETUNING_HELPERS:
        assert name in exported, f"{name} missing from __all__"


def test_aggregation_lives_in_submodule_not_root():
    # Aggregation is reachable from its submodule ...
    aggregation = importlib.import_module("alphagenome_pytorch.aggregation")
    assert hasattr(aggregation, "aggregate_genes")
    # ... and deliberately absent from the package root.
    assert "aggregate_genes" not in alphagenome_pytorch.__all__
    assert not hasattr(alphagenome_pytorch, "aggregate_genes")

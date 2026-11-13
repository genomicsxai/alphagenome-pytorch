"""Unit tests for fine-tune organism normalization, resolution, and selection.

Covers the pieces that make the fine-tune organism authoritative at inference:
``alphagenome_pytorch.organisms`` (strict normalizers) and the
``checkpointing`` context/resolver/selector/finalizer. Pure logic only — no model
weights required.

Run: ``pytest tests/unit/test_organism_resolution.py -v`` (or ``-k organism``).
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from alphagenome_pytorch.organisms import (
    ORGANISM_ALIASES,
    normalize_organism_index,
    normalize_organism_indices,
)
from alphagenome_pytorch.extensions.finetuning.checkpointing import (
    FinetunedOrganismContext,
    finalize_finetuned_organism_context,
    resolve_finetuned_organism,
    select_organism_index,
)


# ---------------------------------------------------------------------------
# normalize_organism_index
# ---------------------------------------------------------------------------

def test_normalize_index_none_is_none():
    assert normalize_organism_index(None, num_organisms=2) is None


@pytest.mark.parametrize("value,expected", [
    ("human", 0), ("Human", 0), ("homo_sapiens", 0),
    ("mouse", 1), ("MOUSE", 1), ("mus_musculus", 1),
    (0, 0), (1, 1),
    ("0", 0), ("1", 1),          # decimal strings from CSV/TSV metadata
])
def test_normalize_index_valid(value, expected):
    assert normalize_organism_index(value, num_organisms=2) == expected


def test_normalize_index_accepts_numpy_int():
    assert normalize_organism_index(np.int64(1), num_organisms=2) == 1


@pytest.mark.parametrize("bad", [True, False])
def test_normalize_index_rejects_bool(bad):
    with pytest.raises(ValueError):
        normalize_organism_index(bad, num_organisms=2)


@pytest.mark.parametrize("bad", ["rat", "dog", "", "  ", 1.0, np.float64(1.0), object()])
def test_normalize_index_rejects_garbage(bad):
    with pytest.raises(ValueError):
        normalize_organism_index(bad, num_organisms=2)


@pytest.mark.parametrize("bad", [2, "2", -1, "-1"])
def test_normalize_index_bounds(bad):
    # Out of range for a 2-organism model.
    with pytest.raises(ValueError):
        normalize_organism_index(bad, num_organisms=2)


def test_normalize_index_third_organism_valid_with_larger_bound():
    # A future three-organism model: index 2 is valid, 3 is not.
    assert normalize_organism_index(2, num_organisms=3) == 2
    with pytest.raises(ValueError):
        normalize_organism_index(3, num_organisms=3)


# ---------------------------------------------------------------------------
# normalize_organism_indices
# ---------------------------------------------------------------------------

def test_normalize_indices_none():
    assert normalize_organism_indices(None, num_organisms=2) is None


def test_normalize_indices_string_is_scalar():
    assert normalize_organism_indices("mouse", num_organisms=2) == (1,)


def test_normalize_indices_dedupe_and_sort():
    assert normalize_organism_indices([1, 0, 1], num_organisms=2) == (0, 1)
    assert normalize_organism_indices({1, 0}, num_organisms=2) == (0, 1)


def test_normalize_indices_empty_raises():
    with pytest.raises(ValueError):
        normalize_organism_indices([], num_organisms=2)


def test_normalize_indices_invalid_member_raises():
    with pytest.raises(ValueError):
        normalize_organism_indices(["mouse", "rat"], num_organisms=2)


# ---------------------------------------------------------------------------
# resolve_finetuned_organism
# ---------------------------------------------------------------------------

def test_resolve_mouse_scalar():
    ctx = resolve_finetuned_organism(checkpoint_organism="mouse", num_organisms=2)
    assert ctx == FinetunedOrganismContext(
        organism_indices=(1,), default_organism_index=1, source="checkpoint",
    )


def test_resolve_human_scalar():
    ctx = resolve_finetuned_organism(checkpoint_organism="human", num_organisms=2)
    assert ctx.organism_indices == (0,) and ctx.default_organism_index == 0


def test_resolve_legacy_fallback():
    ctx = resolve_finetuned_organism(num_organisms=2)
    assert ctx.organism_indices is None
    assert ctx.default_organism_index == 0
    assert ctx.source == "fallback"


def test_resolve_from_track_metadata_only():
    ctx = resolve_finetuned_organism(
        track_metadata=[{"organism": 1}, {"organism": 1}, {"foo": "bar"}],
        num_organisms=2,
    )
    assert ctx.organism_indices == (1,)
    assert ctx.default_organism_index == 1
    assert ctx.source == "track_metadata"


def test_resolve_organism_less_rows_are_no_evidence():
    ctx = resolve_finetuned_organism(
        track_metadata=[{"foo": 1}, {"strand": "+"}], num_organisms=2,
    )
    assert ctx.source == "fallback"
    assert ctx.default_organism_index == 0


def test_resolve_multi_organism_metadata_has_no_default():
    ctx = resolve_finetuned_organism(
        track_metadata=[{"organism": 0}, {"organism": 1}], num_organisms=2,
    )
    assert set(ctx.organism_indices) == {0, 1}
    assert ctx.default_organism_index is None


def test_resolve_plural_precedence_over_scalar():
    ctx = resolve_finetuned_organism(
        organism_indices=[0, 1], checkpoint_organism="mouse", num_organisms=2,
    )
    assert set(ctx.organism_indices) == {0, 1}
    assert ctx.default_organism_index is None
    # scalar "mouse" IS in the plural set -> no conflict
    assert ctx.conflicts == ()


def test_resolve_plural_scalar_conflict_noted_once():
    ctx = resolve_finetuned_organism(
        organism_indices=[0], checkpoint_organism="mouse", num_organisms=2,
    )
    assert ctx.organism_indices == (0,)
    assert ctx.default_organism_index == 0
    assert len(ctx.conflicts) == 1


def test_resolve_checkpoint_authoritative_over_broader_catalog():
    # checkpoint says mouse; catalog lists both -> checkpoint wins, one conflict note.
    ctx = resolve_finetuned_organism(
        checkpoint_organism="mouse",
        track_metadata=[{"organism": 0}, {"organism": 1}],
        num_organisms=2,
    )
    assert ctx.organism_indices == (1,)
    assert ctx.default_organism_index == 1
    assert ctx.source == "checkpoint"
    assert len(ctx.conflicts) == 1


def test_resolve_invalid_value_raises():
    with pytest.raises(ValueError):
        resolve_finetuned_organism(checkpoint_organism="rat", num_organisms=2)


# ---------------------------------------------------------------------------
# select_organism_index
# ---------------------------------------------------------------------------

def test_select_uses_default_when_no_explicit():
    ctx = resolve_finetuned_organism(checkpoint_organism="mouse", num_organisms=2)
    assert select_organism_index(ctx, num_organisms=2) == 1


def test_select_explicit_overrides_and_warns_vs_single_species():
    ctx = resolve_finetuned_organism(checkpoint_organism="mouse", num_organisms=2)
    with pytest.warns(UserWarning):
        idx = select_organism_index(ctx, explicit=0, num_organisms=2)
    assert idx == 0  # explicit forwarded despite the warning


def test_select_fallback_explicit_does_not_warn(recwarn):
    ctx = resolve_finetuned_organism(num_organisms=2)  # fallback, organism_indices=None
    idx = select_organism_index(ctx, explicit=1, num_organisms=2)
    assert idx == 1
    assert len(recwarn) == 0  # unknown provenance -> no "untrained" warning


def test_select_mixed_without_explicit_raises():
    ctx = resolve_finetuned_organism(organism_indices=[0, 1], num_organisms=2)
    with pytest.raises(ValueError):
        select_organism_index(ctx, num_organisms=2)


def test_select_mixed_with_explicit_ok():
    ctx = resolve_finetuned_organism(organism_indices=[0, 1], num_organisms=2)
    assert select_organism_index(ctx, explicit=1, num_organisms=2) == 1


def test_select_explicit_out_of_bounds_raises():
    ctx = resolve_finetuned_organism(checkpoint_organism="mouse", num_organisms=2)
    with pytest.raises(ValueError):
        select_organism_index(ctx, explicit=2, num_organisms=2)


# ---------------------------------------------------------------------------
# finalize_finetuned_organism_context
# ---------------------------------------------------------------------------

def _stub_model(num_organisms: int = 2):
    return types.SimpleNamespace(num_organisms=num_organisms)


def test_finalize_attaches_context_and_writes_metadata():
    model = _stub_model()
    meta = {"organism": "mouse", "track_metadata": None}
    ctx = finalize_finetuned_organism_context(model, meta)
    assert model.finetuned_organism_context is ctx
    assert meta["organism_indices"] == (1,)
    assert meta["default_organism_index"] == 1
    assert meta["organism_resolution_source"] == "checkpoint"


def test_finalize_metadata_and_attribute_agree():
    model = _stub_model()
    meta = {"organism_indices": [1]}
    finalize_finetuned_organism_context(model, meta)
    ctx = model.finetuned_organism_context
    assert meta["organism_indices"] == ctx.organism_indices
    assert meta["default_organism_index"] == ctx.default_organism_index


def test_finalize_legacy_checkpoint_defaults_to_human():
    model = _stub_model()
    meta: dict = {}
    finalize_finetuned_organism_context(model, meta)
    assert meta["default_organism_index"] == 0
    assert meta["organism_resolution_source"] == "fallback"


def test_finalize_emits_conflict_warning_once():
    model = _stub_model()
    meta = {"organism_indices": [0], "organism": "mouse"}
    with pytest.warns(UserWarning) as record:
        finalize_finetuned_organism_context(model, meta)
    assert len(record) == 1


# ---------------------------------------------------------------------------
# export_delta_weights validation (fails before writing a conflicting artifact)
# ---------------------------------------------------------------------------

def test_export_rejects_scalar_not_in_plural(tmp_path):
    from alphagenome_pytorch.extensions.finetuning.checkpointing import (
        export_delta_weights,
    )
    # organism="mouse" (1) is not in organism_indices=[0] -> must raise up front,
    # before any weights are extracted or written.
    with pytest.raises(ValueError):
        export_delta_weights(
            _stub_model(), config=None, path=tmp_path / "x.safetensors",
            organism="mouse", organism_indices=[0],
        )
    assert not (tmp_path / "x.safetensors").exists()

"""Unit tests for ServedModelRouter swap mechanics + catalog parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from alphagenome_pytorch.extensions.finetuning.adapters import LoRA
from alphagenome_pytorch.extensions.serving.router import (
    AmbiguousModelError,
    CatalogAdapterSpec,
    CatalogBaseSpec,
    ModelNotFoundError,
    ServedModelEntry,
    ServedModelRouter,
    _AdapterAttachment,
    capture_adapter_attachments,
    load_catalog,
)


# ---------------------------------------------------------------------------
# Synthetic model fixture
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Stand-in for ``AlphaGenome``: just enough surface for the router."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.heads = nn.ModuleDict()


def _make_lora_attachment(model: _TinyModel) -> _AdapterAttachment:
    """Wrap ``model.q_proj`` in a LoRA module and return the attachment record."""
    original = model.q_proj
    wrapper = LoRA(original, rank=2)
    return _AdapterAttachment(parent=model, attr_name="q_proj", wrapper=wrapper)


def _make_entry(
    model: _TinyModel,
    *,
    id: str,
    kind: str = "adapter",
    head_module: nn.Module | None = None,
) -> ServedModelEntry:
    """Build a hand-rolled entry without going through prepare_for_transfer."""
    attachments: list[_AdapterAttachment] = []
    head_modules: dict[str, nn.Module] = {}
    if kind == "adapter":
        attachments.append(_make_lora_attachment(model))
    if head_module is not None:
        head_modules[id] = head_module
    return ServedModelEntry(
        id=id, label=id.upper(), kind=kind,
        base_model_hash="sha256:demo",
        adapter_attachments=attachments,
        head_modules=head_modules,
    )


def _stub_runtime() -> Any:
    @dataclass
    class _Stub:
        sequence_source: Any = None
        metadata_catalog: Any = None
        device: torch.device = torch.device("cpu")
    return _Stub()


# ---------------------------------------------------------------------------
# capture_adapter_attachments
# ---------------------------------------------------------------------------


class TestCaptureAttachments:
    def test_no_adapters(self) -> None:
        m = _TinyModel()
        assert capture_adapter_attachments(m) == []

    def test_finds_lora(self) -> None:
        m = _TinyModel()
        m.q_proj = LoRA(m.q_proj, rank=2)
        atts = capture_adapter_attachments(m)
        assert len(atts) == 1
        assert atts[0].parent is m
        assert atts[0].attr_name == "q_proj"
        assert isinstance(atts[0].wrapper, LoRA)

    def test_does_not_recurse_into_wrappers(self) -> None:
        # Even though LoRA stores .original_layer which is a Linear, the
        # walker must not yield the inner layer as a separate attachment.
        m = _TinyModel()
        m.q_proj = LoRA(m.q_proj, rank=2)
        atts = capture_adapter_attachments(m)
        assert len(atts) == 1


# ---------------------------------------------------------------------------
# Router selection / swap
# ---------------------------------------------------------------------------


class TestRouterSelection:
    def test_resolve_singleton_no_id(self) -> None:
        m = _TinyModel()
        entry = _make_entry(m, id="only", kind="base")
        r = ServedModelRouter(base_model=m, runtime=_stub_runtime(), entries=[entry])
        assert r.resolve_model_id(None) == "only"

    def test_resolve_multi_no_id_raises(self) -> None:
        m = _TinyModel()
        e1 = _make_entry(m, id="a", kind="base")
        e2 = _make_entry(m, id="b", kind="base")
        r = ServedModelRouter(base_model=m, runtime=_stub_runtime(), entries=[e1, e2])
        with pytest.raises(AmbiguousModelError):
            r.resolve_model_id(None)

    def test_resolve_unknown_id_raises(self) -> None:
        m = _TinyModel()
        entry = _make_entry(m, id="only", kind="base")
        r = ServedModelRouter(base_model=m, runtime=_stub_runtime(), entries=[entry])
        with pytest.raises(ModelNotFoundError):
            r.resolve_model_id("nope")

    def test_duplicate_ids_in_constructor_raise(self) -> None:
        m = _TinyModel()
        e1 = _make_entry(m, id="dup", kind="base")
        e2 = _make_entry(m, id="dup", kind="base")
        with pytest.raises(ValueError, match="Duplicate"):
            ServedModelRouter(base_model=m, runtime=_stub_runtime(), entries=[e1, e2])

    def test_empty_entries_raise(self) -> None:
        m = _TinyModel()
        with pytest.raises(ValueError, match="at least one"):
            ServedModelRouter(base_model=m, runtime=_stub_runtime(), entries=[])

    def test_list_models(self) -> None:
        m = _TinyModel()
        e1 = _make_entry(m, id="base", kind="base")
        e2 = _make_entry(m, id="lora-1", kind="adapter")
        r = ServedModelRouter(base_model=m, runtime=_stub_runtime(), entries=[e1, e2])
        rows = r.list_models()
        assert [r["id"] for r in rows] == ["base", "lora-1"]
        assert rows[0]["kind"] == "base"
        assert rows[1]["kind"] == "adapter"


# ---------------------------------------------------------------------------
# Swap mechanics on a tiny model
# ---------------------------------------------------------------------------


class TestSwapMechanics:
    def test_attach_then_detach_restores_original(self) -> None:
        m = _TinyModel()
        original = m.q_proj  # save a reference
        e = _make_entry(m, id="lora", kind="adapter")
        # Sanity: entry was built clean — base is not currently wrapped.
        assert m.q_proj is original

        r = ServedModelRouter(base_model=m, runtime=_stub_runtime(), entries=[e])
        # Bypass the LocalDnaModelAdapter construction — just exercise the
        # private attach/detach pair by selecting and then detaching manually.
        r._attach_locked(e)
        r._active_id = "lora"
        assert isinstance(m.q_proj, LoRA)
        assert m.q_proj.original_layer is original

        r._detach_active_locked()
        assert m.q_proj is original
        assert r._active_id is None

    def test_swap_between_two_entries(self) -> None:
        m = _TinyModel()
        # Build two entries against the same base — each wraps q_proj
        # independently with its own LoRA module.
        e1 = _make_entry(m, id="a", kind="adapter")
        e2 = _make_entry(m, id="b", kind="adapter")

        # Both wrappers stored their original_layer reference at construction
        # time; both should equal the bare Linear.
        original = m.q_proj
        assert e1.adapter_attachments[0].wrapper.original_layer is original
        assert e2.adapter_attachments[0].wrapper.original_layer is original

        r = ServedModelRouter(base_model=m, runtime=_stub_runtime(), entries=[e1, e2])

        r._attach_locked(e1)
        r._active_id = "a"
        assert m.q_proj is e1.adapter_attachments[0].wrapper

        # Swap.
        r._detach_active_locked()
        r._attach_locked(e2)
        r._active_id = "b"
        assert m.q_proj is e2.adapter_attachments[0].wrapper

        # Detach again — base is back to Linear.
        r._detach_active_locked()
        assert m.q_proj is original

    def test_head_attach_detach(self) -> None:
        m = _TinyModel()
        head = nn.Linear(8, 4)
        e = ServedModelEntry(
            id="with_head", label=None, kind="adapter",
            base_model_hash="sha256:demo",
            adapter_attachments=[],  # no adapter — just a new head.
            head_modules={"new_head": head},
        )
        r = ServedModelRouter(base_model=m, runtime=_stub_runtime(), entries=[e])
        r._attach_locked(e)
        r._active_id = "with_head"
        assert m.heads["new_head"] is head
        r._detach_active_locked()
        assert "new_head" not in m.heads


# ---------------------------------------------------------------------------
# Catalog file parsing
# ---------------------------------------------------------------------------


class TestCatalogLoading:
    def test_yaml_full(self, tmp_path: Path) -> None:
        p = tmp_path / "catalog.yaml"
        p.write_text(
            "base:\n"
            "  id: ag-base\n"
            "  label: AlphaGenome (base)\n"
            "adapters:\n"
            "  - id: lora-1\n"
            "    source: local:/srv/lora-1\n"
            "    label: First LoRA\n"
            "  - id: lora-2\n"
            "    source: hf://org/repo@v1\n"
        )
        spec = load_catalog(p)
        assert spec.base.enabled
        assert spec.base.id == "ag-base"
        assert spec.base.label == "AlphaGenome (base)"
        assert [a.id for a in spec.adapters] == ["lora-1", "lora-2"]
        assert spec.adapters[0].source == "local:/srv/lora-1"
        assert spec.adapters[0].label == "First LoRA"

    def test_json_catalog(self, tmp_path: Path) -> None:
        p = tmp_path / "catalog.json"
        p.write_text(json.dumps({
            "adapters": [
                {"id": "x", "source": "local:/x"},
            ],
        }))
        spec = load_catalog(p)
        assert not spec.base.enabled
        assert spec.adapters[0].id == "x"

    def test_missing_id_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "catalog.yaml"
        p.write_text("adapters:\n  - source: local:/x\n")
        with pytest.raises(ValueError, match="must have 'id' and 'source'"):
            load_catalog(p)

    def test_duplicate_ids_raise(self, tmp_path: Path) -> None:
        p = tmp_path / "catalog.yaml"
        p.write_text(
            "base:\n  id: shared\n"
            "adapters:\n"
            "  - id: shared\n"
            "    source: local:/x\n"
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_catalog(p)

    def test_top_level_must_be_mapping(self, tmp_path: Path) -> None:
        p = tmp_path / "catalog.yaml"
        p.write_text("- not: a mapping\n")
        with pytest.raises(ValueError, match="mapping"):
            load_catalog(p)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_catalog(tmp_path / "nope.yaml")

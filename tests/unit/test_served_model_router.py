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

    def test_same_name_head_overlays_and_restores_native_head(self) -> None:
        m = _TinyModel()
        native_head = nn.Linear(8, 8)
        adapter_head = nn.Linear(8, 4)
        m.heads["atac"] = native_head
        entry = ServedModelEntry(
            id="finetuned", label=None, kind="adapter",
            base_model_hash="sha256:demo",
            head_modules={"atac": adapter_head},
        )
        router = ServedModelRouter(
            base_model=m, runtime=_stub_runtime(), entries=[entry]
        )

        router._attach_locked(entry)
        router._active_id = entry.id
        assert m.heads["atac"] is adapter_head

        router._detach_active_locked()
        assert m.heads["atac"] is native_head

    def test_two_entries_can_overlay_the_same_head_name(self) -> None:
        m = _TinyModel()
        native_head = nn.Linear(8, 8)
        head_a = nn.Linear(8, 4)
        head_b = nn.Linear(8, 6)
        m.heads["atac"] = native_head
        base = _make_entry(m, id="base", kind="base")
        entry_a = ServedModelEntry(
            id="a", label=None, kind="adapter",
            base_model_hash="sha256:demo",
            head_modules={"atac": head_a},
        )
        entry_b = ServedModelEntry(
            id="b", label=None, kind="adapter",
            base_model_hash="sha256:demo",
            head_modules={"atac": head_b},
        )
        router = ServedModelRouter(
            base_model=m,
            runtime=_stub_runtime(),
            entries=[base, entry_a, entry_b],
            adapter_factory=lambda _router, entry: entry.id,
        )

        assert router.select("a") == "a"
        assert m.heads["atac"] is head_a
        assert router.select("b") == "b"
        assert m.heads["atac"] is head_b
        assert router.select("base") == "base"
        assert m.heads["atac"] is native_head


# ---------------------------------------------------------------------------
# Request-scoped locking (acquire)
# ---------------------------------------------------------------------------


class TestAcquireLocking:
    """``acquire`` must hold the lock across the swap *and* the model call so a
    concurrent request cannot re-swap the shared base model mid-forward."""

    def _router(self) -> ServedModelRouter:
        m = _TinyModel()
        e1 = _make_entry(m, id="a", kind="adapter")
        e2 = _make_entry(m, id="b", kind="adapter")
        return ServedModelRouter(
            base_model=m, runtime=_stub_runtime(), entries=[e1, e2],
            adapter_factory=lambda router, entry: entry.id,
        )

    def test_acquire_yields_active_adapter(self) -> None:
        r = self._router()
        with r.acquire("a") as adapter:
            assert adapter == "a"
            assert r.active_id == "a"

    def test_acquire_blocks_second_thread_until_release(self) -> None:
        import threading

        r = self._router()
        a_held = threading.Event()
        release_a = threading.Event()
        b_entered = threading.Event()
        active_while_a_held: list[str | None] = []

        def hold_a() -> None:
            with r.acquire("a"):
                a_held.set()
                # Block inside the critical section, simulating a long forward.
                release_a.wait(timeout=5)
                # Record what B managed to do while we held the lock.
                active_while_a_held.append(r.active_id)

        def take_b() -> None:
            a_held.wait(timeout=5)
            with r.acquire("b"):
                b_entered.set()

        t1 = threading.Thread(target=hold_a)
        t2 = threading.Thread(target=take_b)
        t1.start()
        assert a_held.wait(timeout=5)
        t2.start()

        # B is blocked on the router lock: it must not swap in while A holds it.
        assert not b_entered.wait(timeout=0.5)
        assert r.active_id == "a"

        # Release A; B can now proceed and become active.
        release_a.set()
        assert b_entered.wait(timeout=5)
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert r.active_id == "b"
        # While A held the lock, the active model never changed to B.
        assert active_while_a_held == ["a"]

    def test_default_factory_forwards_resolved_default_organism(
        self, monkeypatch
    ) -> None:
        """The default factory must pass a resolved ``entry.default_organism`` to
        the per-request runtime (so a mouse bundle does not default to human),
        and omit it when unresolved (mixed/None)."""
        import alphagenome_pytorch.prediction as prediction_mod
        from alphagenome_pytorch.extensions.serving.router import (
            _default_adapter_factory,
        )

        captured: dict[str, Any] = {}

        class _FakeRuntime:
            def __init__(self, **kwargs: Any) -> None:
                captured.clear()
                captured.update(kwargs)

        monkeypatch.setattr(
            prediction_mod, "AlphaGenomePredictionRuntime", _FakeRuntime
        )

        m = _TinyModel()
        mouse = _make_entry(m, id="mouse", kind="adapter")
        mouse.default_organism = 1
        r = ServedModelRouter(base_model=m, runtime=_stub_runtime(), entries=[mouse])
        _default_adapter_factory(r, mouse)
        assert captured.get("default_organism") == 1

        human_default_absent = _make_entry(m, id="mixed", kind="adapter")
        human_default_absent.default_organism = None
        _default_adapter_factory(r, human_default_absent)
        assert "default_organism" not in captured

    def test_default_factory_shares_entry_runtime_with_scorer(self) -> None:
        """When an entry carries a pre-built runtime, the service adapter and its
        scorer must share it — so predict and score resolve organism/metadata
        identically (a mouse entry defaults to mouse on both paths)."""
        from types import SimpleNamespace

        from tests.unit.serving_fakes import FakeRuntime

        m = _TinyModel()
        rt = FakeRuntime()
        rt.default_organism_index = 1  # mouse
        catalog = object()
        rt.metadata_catalog = catalog
        scorer = SimpleNamespace(runtime=rt)  # scorer bound to the same runtime

        entry = _make_entry(m, id="mouse")
        entry.runtime = rt
        entry.scorer = scorer
        entry.metadata_catalog = catalog

        r = ServedModelRouter(base_model=m, runtime=_stub_runtime(), entries=[entry])
        adapter = r.select("mouse")  # exercises the default factory

        assert adapter.runtime is adapter.scorer.runtime
        assert adapter.runtime.metadata_catalog is entry.metadata_catalog
        assert adapter.runtime.resolve_organism_index(None) == 1


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

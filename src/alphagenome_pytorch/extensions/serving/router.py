"""Catalog-mode serving: shared base trunk + hot-swap finetune adapters.

The router holds one resident base ``AlphaGenome`` model and a registry of
:class:`ServedModelEntry` objects. Each entry captures the wrapper modules
(``LoRA`` / ``Locon`` / ``IA3`` / ``Houlsby*``) and new heads produced by
applying a finetune's :class:`TransferConfig` to the base. Entries share the
underlying frozen trunk parameters by reference — the wrappers' ``original_layer``
attributes point at the same ``Linear``/``Conv1d`` instances inside ``base_model``.

A :meth:`ServedModelRouter.select` call detaches whichever entry is currently
active (``setattr`` the wrappers back to their ``original_layer``; remove
new heads from ``base_model.heads``) and attaches the requested entry. Swap
is serialized behind a lock; v1 makes no attempt to overlap requests.

Catalog mode is REST-first: gRPC users must pass ``alphagenome-model-id``
metadata explicitly (no default fallback). Constraints:

- ``--no-merge-adapters`` semantics permanently. Merged adapters are
  irreversible; the router rejects any entry whose ``transfer_config.mode``
  ended up merged.
- No ``torch.compile``. Swap mutates the live module tree.
- v1 forbids entries that use ``keep_heads`` / ``remove_heads`` (would
  permanently mutate the base ``heads`` dict). Trivial finetune configs
  (default ``new_heads=...``) work fine.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from alphagenome_pytorch.extensions.serving.adapter import LocalDnaModelAdapter
from alphagenome_pytorch.extensions.serving.bundle import BundlePaths, Manifest


# ---------------------------------------------------------------------------
# Adapter detection
# ---------------------------------------------------------------------------


def _adapter_classes() -> tuple[type, ...]:
    """Lazy import of adapter wrapper types (avoids hard dependency on finetuning)."""
    from alphagenome_pytorch.extensions.finetuning.adapters import (
        IA3,
        IA3_FF,
        HoulsbyBlockWrapper,
        HoulsbyWrapper,
        LoRA,
        Locon,
    )
    return (LoRA, Locon, IA3, IA3_FF, HoulsbyWrapper, HoulsbyBlockWrapper)


@dataclass
class _AdapterAttachment:
    """One (parent, attr_name, wrapper) tuple captured from the model tree."""

    parent: nn.Module
    attr_name: str
    wrapper: nn.Module  # has `.original_layer` (LoRA/Locon/IA3/HoulsbyWrapper)
                         # or `.block` (HoulsbyBlockWrapper)


def _wrapped_target(wrapper: nn.Module) -> nn.Module:
    """Return the inner layer/block a wrapper restores on detach."""
    if hasattr(wrapper, "original_layer"):
        return wrapper.original_layer
    if hasattr(wrapper, "block"):
        return wrapper.block
    raise TypeError(
        f"Cannot detach {type(wrapper).__name__}: no original_layer/block attribute"
    )


def capture_adapter_attachments(model: nn.Module) -> list[_AdapterAttachment]:
    """Walk the module tree and capture every adapter wrapper as
    (parent, attr_name, wrapper)."""
    out: list[_AdapterAttachment] = []
    adapter_types = _adapter_classes()

    def visit(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, adapter_types):
                out.append(_AdapterAttachment(parent=parent, attr_name=name, wrapper=child))
                # Do NOT recurse into adapter wrappers — their inner
                # original_layer is the unwrapped trunk module.
                continue
            visit(child)

    visit(model)
    return out


# ---------------------------------------------------------------------------
# Served model entries
# ---------------------------------------------------------------------------


@dataclass
class ServedModelEntry:
    """A model addressable by ``model_id`` in catalog-mode serving.

    Entries are pre-built at startup. ``select`` swaps an entry's wrappers
    onto the shared ``base_model`` (no weight copy — wrappers hold references
    to the trunk).
    """

    id: str
    label: str | None
    kind: str  # "base" | "adapter"
    base_model_hash: str
    adapter_attachments: list[_AdapterAttachment] = field(default_factory=list)
    head_modules: dict[str, nn.Module] = field(default_factory=dict)
    metadata_catalog: Any | None = None
    track_names: Any | None = None
    scorer: Any | None = None
    manifest: Manifest | None = None

    def is_base(self) -> bool:
        return self.kind == "base"


# ---------------------------------------------------------------------------
# Building entries from bundles
# ---------------------------------------------------------------------------


def _ensure_swappable_config(transfer_config: Any) -> None:
    if getattr(transfer_config, "keep_heads", None):
        raise ValueError(
            "Catalog mode does not support adapters that use keep_heads; the "
            "router cannot reversibly remove base heads. Run such adapters in "
            "singleton mode."
        )
    if getattr(transfer_config, "remove_heads", None):
        raise ValueError(
            "Catalog mode does not support adapters that use remove_heads; "
            "the router cannot reversibly remove base heads. Run such "
            "adapters in singleton mode."
        )


def build_adapter_entry(
    *,
    base_model: nn.Module,
    bundle_paths: BundlePaths,
    manifest: Manifest,
    metadata_catalog: Any | None = None,
    track_names: Any | None = None,
    scorer: Any | None = None,
) -> ServedModelEntry:
    """Build a :class:`ServedModelEntry` for an adapter bundle.

    Mutates ``base_model`` to apply the bundle's ``TransferConfig`` and load
    delta weights, then captures the wrappers + new heads and *detaches* them
    so ``base_model`` is left in its original clean state.
    """
    from alphagenome_pytorch.extensions.finetuning.checkpointing import (
        compute_base_model_hash,
        load_delta_config,
        load_delta_weights,
    )
    from alphagenome_pytorch.extensions.finetuning.transfer import (
        prepare_for_transfer,
    )

    transfer_config = load_delta_config(bundle_paths.adapter_safetensors)
    _ensure_swappable_config(transfer_config)

    actual_hash = compute_base_model_hash(base_model)
    if actual_hash != manifest.base_model_hash:
        raise ValueError(
            f"Bundle {manifest.id!r} declares base_model_hash="
            f"{manifest.base_model_hash!r} but base model hashes to "
            f"{actual_hash!r}. Refusing to load incompatible adapter."
        )

    new_head_names = list(transfer_config.new_heads.keys())

    # Snapshot which heads existed before — anything new that appears must be
    # removed during detach. (prepare_for_transfer only adds when keep_heads /
    # remove_heads are unset, which we enforce above.)
    pre_heads = set(base_model.heads.keys()) if hasattr(base_model, "heads") else set()

    prepare_for_transfer(base_model, transfer_config)
    load_delta_weights(base_model, bundle_paths.adapter_safetensors, strict=False)

    attachments = capture_adapter_attachments(base_model)

    head_modules: dict[str, nn.Module] = {}
    if hasattr(base_model, "heads"):
        post_heads = set(base_model.heads.keys())
        added = post_heads - pre_heads
        # Sanity: declared new heads must match what got added.
        for h in new_head_names:
            if h not in added:
                # Either the head was already there (replaced) or transfer_config
                # disagrees with what prepare_for_transfer did — both are bugs.
                raise ValueError(
                    f"Bundle {manifest.id!r}: declared new head {h!r} was not "
                    "added to base_model.heads during prepare_for_transfer."
                )
            head_modules[h] = base_model.heads[h]
        # Detect unexpected additions — surface as an error rather than silently
        # leaving them attached.
        for h in added - set(new_head_names):
            raise ValueError(
                f"Bundle {manifest.id!r} added unexpected head {h!r}; "
                "transfer_config.new_heads is the source of truth."
            )

    # Detach: restore base_model to clean state.
    for att in attachments:
        setattr(att.parent, att.attr_name, _wrapped_target(att.wrapper))
    if hasattr(base_model, "heads"):
        for h in new_head_names:
            del base_model.heads[h]

    return ServedModelEntry(
        id=manifest.id,
        label=manifest.label or manifest.id,
        kind="adapter",
        base_model_hash=manifest.base_model_hash,
        adapter_attachments=attachments,
        head_modules=head_modules,
        metadata_catalog=metadata_catalog,
        track_names=track_names,
        scorer=scorer,
        manifest=manifest,
    )


def build_base_entry(
    *,
    base_model: nn.Module,
    id: str,
    label: str | None = None,
    metadata_catalog: Any | None = None,
    track_names: Any | None = None,
    scorer: Any | None = None,
) -> ServedModelEntry:
    """Build a :class:`ServedModelEntry` for the bare base model."""
    from alphagenome_pytorch.extensions.finetuning.checkpointing import (
        compute_base_model_hash,
    )
    return ServedModelEntry(
        id=id,
        label=label or id,
        kind="base",
        base_model_hash=compute_base_model_hash(base_model),
        adapter_attachments=[],
        head_modules={},
        metadata_catalog=metadata_catalog,
        track_names=track_names,
        scorer=scorer,
        manifest=None,
    )


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------


def _default_adapter_factory(
    router: "ServedModelRouter", entry: ServedModelEntry
) -> LocalDnaModelAdapter:
    """Default adapter factory: a fresh runtime bound to the live base model."""
    from alphagenome_pytorch.prediction import AlphaGenomePredictionRuntime

    entry_runtime = AlphaGenomePredictionRuntime(
        model=router.base_model,
        sequence_source=getattr(router.runtime, "sequence_source", None),
        metadata_catalog=entry.metadata_catalog
        or getattr(router.runtime, "metadata_catalog", None),
        track_names=entry.track_names,
        device=getattr(router.runtime, "device", None),
    )
    return LocalDnaModelAdapter(entry_runtime, scorer=entry.scorer)


class ModelNotFoundError(KeyError):
    """Raised when a request references a model_id not in the catalog."""


class AmbiguousModelError(ValueError):
    """Raised when a catalog has >1 model and the request gives no model_id."""


class ServedModelRouter:
    """Owns the shared base model and routes per-request to an entry's adapter."""

    def __init__(
        self,
        *,
        base_model: nn.Module,
        runtime: Any,
        entries: list[ServedModelEntry],
        adapter_factory: Any | None = None,
    ) -> None:
        """Construct a router.

        ``adapter_factory`` (optional) is called as
        ``adapter_factory(router, entry) -> LocalDnaModelAdapter`` whenever a
        request selects ``entry``. Defaults to building a fresh
        :class:`AlphaGenomePredictionRuntime` bound to the live (active) base
        model. Tests can inject a fake to avoid model construction overhead.
        """
        if not entries:
            raise ValueError("ServedModelRouter requires at least one entry.")
        seen: set[str] = set()
        self._entries: dict[str, ServedModelEntry] = {}
        self._order: list[str] = []
        for e in entries:
            if e.id in seen:
                raise ValueError(f"Duplicate model id in catalog: {e.id!r}")
            seen.add(e.id)
            self._entries[e.id] = e
            self._order.append(e.id)
        self.base_model = base_model
        self.runtime = runtime
        self._adapter_factory = adapter_factory or _default_adapter_factory
        self._active_id: str | None = None
        self._lock = threading.RLock()

    # ---- public surface --------------------------------------------------

    @property
    def model_ids(self) -> list[str]:
        return list(self._order)

    @property
    def active_id(self) -> str | None:
        return self._active_id

    def has(self, model_id: str) -> bool:
        return model_id in self._entries

    def get_entry(self, model_id: str) -> ServedModelEntry:
        try:
            return self._entries[model_id]
        except KeyError as exc:
            raise ModelNotFoundError(model_id) from exc

    def list_models(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for mid in self._order:
            e = self._entries[mid]
            row: dict[str, Any] = {
                "id": e.id,
                "label": e.label,
                "kind": e.kind,
                "base_model_hash": e.base_model_hash,
            }
            if e.manifest is not None:
                row["genome"] = e.manifest.genome
                row["organism"] = e.manifest.organism
                row["modality"] = e.manifest.modality
                row["biosample"] = e.manifest.biosample
                if e.manifest.adapter_summary:
                    row["adapter"] = e.manifest.adapter_summary
            out.append(row)
        return out

    def resolve_model_id(self, requested: str | None) -> str:
        """Pick the model_id for a request; raise on ambiguity / unknown id."""
        if requested is None:
            if len(self._entries) == 1:
                return self._order[0]
            raise AmbiguousModelError(
                "Multiple models registered; request must specify a model_id. "
                f"Available: {self._order}"
            )
        if requested not in self._entries:
            raise ModelNotFoundError(requested)
        return requested

    def select(self, model_id: str) -> LocalDnaModelAdapter:
        """Activate ``model_id`` on the shared base and return a service adapter.

        Serialized: callers must hold the result while making prediction calls.
        """
        with self._lock:
            if model_id not in self._entries:
                raise ModelNotFoundError(model_id)
            if self._active_id != model_id:
                self._detach_active_locked()
                self._attach_locked(self._entries[model_id])
                self._active_id = model_id
            entry = self._entries[model_id]
            return self._adapter_factory(self, entry)

    # ---- swap mechanics --------------------------------------------------

    def _detach_active_locked(self) -> None:
        if self._active_id is None:
            return
        entry = self._entries[self._active_id]
        for att in entry.adapter_attachments:
            setattr(att.parent, att.attr_name, _wrapped_target(att.wrapper))
        if hasattr(self.base_model, "heads"):
            for hname in entry.head_modules:
                if hname in self.base_model.heads:
                    del self.base_model.heads[hname]
        self._active_id = None

    def _attach_locked(self, entry: ServedModelEntry) -> None:
        if entry.is_base():
            return
        for att in entry.adapter_attachments:
            setattr(att.parent, att.attr_name, att.wrapper)
        if hasattr(self.base_model, "heads"):
            for hname, hmod in entry.head_modules.items():
                self.base_model.heads[hname] = hmod


# ---------------------------------------------------------------------------
# Catalog file format
# ---------------------------------------------------------------------------


@dataclass
class CatalogBaseSpec:
    id: str = "alphagenome-base"
    label: str | None = None
    enabled: bool = False


@dataclass
class CatalogAdapterSpec:
    id: str
    source: str
    label: str | None = None


@dataclass
class CatalogSpec:
    base: CatalogBaseSpec
    adapters: list[CatalogAdapterSpec]


def load_catalog(path: str | Path) -> CatalogSpec:
    """Load a catalog YAML or JSON file describing a set of served models.

    Schema::

        base:
          id: alphagenome-base       # optional; if present, base is also served
          label: AlphaGenome (base)
        adapters:
          - id: wtc11-atac-lora
            source: hf://your-org/alphagenome-wtc11-atac-lora
          - id: k562-rna-locon
            source: local:/srv/bundles/k562-rna-locon
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Catalog file not found: {p}")

    text = p.read_text()
    if p.suffix.lower() == ".json":
        import json
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "Reading YAML catalog files requires PyYAML. Install with "
                "'pip install alphagenome-pytorch[serving]' or use a .json catalog."
            ) from exc
        data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError(f"Catalog file {p}: top level must be a mapping")

    base_data = data.get("base") or {}
    if not isinstance(base_data, dict):
        raise ValueError(f"Catalog file {p}: 'base' must be a mapping")
    base = CatalogBaseSpec(
        id=base_data.get("id", "alphagenome-base"),
        label=base_data.get("label"),
        enabled=bool(base_data),
    )

    adapters_data = data.get("adapters") or []
    if not isinstance(adapters_data, list):
        raise ValueError(f"Catalog file {p}: 'adapters' must be a list")

    adapters: list[CatalogAdapterSpec] = []
    seen_ids: set[str] = set()
    if base.enabled:
        seen_ids.add(base.id)
    for i, item in enumerate(adapters_data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Catalog file {p}: adapters[{i}] must be a mapping"
            )
        if "id" not in item or "source" not in item:
            raise ValueError(
                f"Catalog file {p}: adapters[{i}] must have 'id' and 'source'"
            )
        if item["id"] in seen_ids:
            raise ValueError(
                f"Catalog file {p}: duplicate model id {item['id']!r}"
            )
        seen_ids.add(item["id"])
        adapters.append(CatalogAdapterSpec(
            id=str(item["id"]),
            source=str(item["source"]),
            label=item.get("label"),
        ))

    return CatalogSpec(base=base, adapters=adapters)

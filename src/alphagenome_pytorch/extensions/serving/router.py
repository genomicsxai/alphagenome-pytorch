"""Catalog-mode serving: shared base trunk + hot-swap finetune adapters.

The router holds one resident base ``AlphaGenome`` model and a registry of
:class:`ServedModelEntry` objects. Each entry captures the wrapper modules
(``LoRA`` / ``Locon`` / ``IA3`` / ``Houlsby*``) and new heads produced by
applying a finetune's :class:`TransferConfig` to the base. Entries share the
underlying frozen trunk parameters by reference — the wrappers' ``original_layer``
attributes point at the same ``Linear``/``Conv1d`` instances inside ``base_model``.

A :meth:`ServedModelRouter.select` call detaches whichever entry is currently
active (``setattr`` the wrappers back to their ``original_layer``; restore the
native ``base_model.heads`` registry) and attaches the requested entry. Adapter
heads are an overlay on that native registry, so a finetune may use the same
head name as the base or another catalog entry. Swap is serialized behind a
lock; v1 makes no attempt to overlap requests.

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

import contextlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def _restore_head_modules(
    model: nn.Module, native_heads: dict[str, nn.Module]
) -> None:
    """Restore ``model.heads`` exactly to a captured native registry.

    The snapshot stores module references rather than aliasing the mutable
    ``ModuleDict``. This makes same-name finetuned heads a reversible overlay:
    restoring swaps the original module object back into its registry slot.
    """
    if not hasattr(model, "heads"):
        return
    heads = model.heads
    for name in list(heads.keys()):
        del heads[name]
    for name, module in native_heads.items():
        heads[name] = module


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
    default_organism: int | None = None
    # Pre-built runtime shared by this entry's service adapter and its scorer so
    # organism/metadata resolution is identical across predict and score paths.
    runtime: Any | None = None
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
    default_organism: int | None = None,
    runtime: Any | None = None,
    base_model_weights_hash: str | None = None,
) -> ServedModelEntry:
    """Build a :class:`ServedModelEntry` for an adapter bundle.

    Mutates ``base_model`` to apply the bundle's ``TransferConfig`` and load
    delta weights, then captures the wrappers + new heads and *detaches* them
    so ``base_model`` is left in its original clean state. When available,
    ``base_model_weights_hash`` should be computed from the source weights file;
    direct callers that omit it fall back to hashing the live model.
    """
    from alphagenome_pytorch.extensions.finetuning.checkpointing import (
        base_model_structure_hashes_match,
        compute_base_model_hash,
        compute_base_model_weights_hash,
        load_delta_config,
        load_delta_weights,
    )
    from alphagenome_pytorch.extensions.finetuning.transfer import (
        prepare_for_transfer,
    )

    transfer_config = load_delta_config(bundle_paths.adapter_safetensors)
    _ensure_swappable_config(transfer_config)

    actual_hash = compute_base_model_hash(base_model)
    if not base_model_structure_hashes_match(actual_hash, manifest.base_model_hash):
        raise ValueError(
            f"Bundle {manifest.id!r} declares base_model_hash="
            f"{manifest.base_model_hash!r} but base model hashes to "
            f"{actual_hash!r}. Refusing to load incompatible adapter."
        )
    if manifest.base_model_weights_hash is not None:
        actual_weights_hash = base_model_weights_hash
        if actual_weights_hash is None:
            actual_weights_hash = compute_base_model_weights_hash(base_model)
        if actual_weights_hash != manifest.base_model_weights_hash:
            raise ValueError(
                f"Bundle {manifest.id!r} declares base_model_weights_hash="
                f"{manifest.base_model_weights_hash!r} but the supplied base "
                f"hashes to {actual_weights_hash!r}. Refusing to load an adapter "
                "on a different base checkpoint/fold."
            )

    new_head_names = list(transfer_config.new_heads.keys())
    native_heads = (
        dict(base_model.heads.items()) if hasattr(base_model, "heads") else {}
    )
    native_wrapper_ids = {
        id(att.wrapper) for att in capture_adapter_attachments(base_model)
    }
    attachments: list[_AdapterAttachment] = []
    head_modules: dict[str, nn.Module] = {}
    try:
        prepare_for_transfer(base_model, transfer_config)
        attachments = [
            att
            for att in capture_adapter_attachments(base_model)
            if id(att.wrapper) not in native_wrapper_ids
        ]
        load_delta_weights(
            base_model, bundle_paths.adapter_safetensors, strict=False
        )

        if new_head_names and not hasattr(base_model, "heads"):
            raise ValueError(
                f"Bundle {manifest.id!r} declares new heads but the base model "
                "has no heads registry."
            )
        if hasattr(base_model, "heads"):
            missing = [h for h in new_head_names if h not in base_model.heads]
            if missing:
                raise ValueError(
                    f"Bundle {manifest.id!r}: declared heads {missing!r} were not "
                    "present in base_model.heads after prepare_for_transfer."
                )
            head_modules = {
                h: base_model.heads[h] for h in new_head_names
            }

            # Declared heads may replace native registry slots or add new ones,
            # but preparation must not create undeclared entries.
            allowed_heads = set(native_heads) | set(new_head_names)
            unexpected = [
                h for h in base_model.heads if h not in allowed_heads
            ]
            if unexpected:
                raise ValueError(
                    f"Bundle {manifest.id!r} added unexpected heads "
                    f"{unexpected!r}; transfer_config.new_heads is the source "
                    "of truth."
                )

        # prepare_for_transfer builds some adapter params (e.g. IA3 ``scale``)
        # and heads on CPU regardless of the base device. Detached entries do
        # not ride along in a later base_model.to(device), so align them now.
        try:
            device = next(base_model.parameters()).device
        except StopIteration:
            device = None
        if device is not None:
            for att in attachments:
                att.wrapper.to(device)
            for hmod in head_modules.values():
                hmod.to(device)
    finally:
        # Entry construction is transactional: even a partially prepared model
        # is returned to its native wrappers and exact native head registry.
        for att in capture_adapter_attachments(base_model):
            if id(att.wrapper) not in native_wrapper_ids:
                setattr(
                    att.parent,
                    att.attr_name,
                    _wrapped_target(att.wrapper),
                )
        _restore_head_modules(base_model, native_heads)

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
        default_organism=default_organism,
        runtime=runtime,
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
    default_organism: int | None = None,
    runtime: Any | None = None,
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
        default_organism=default_organism,
        runtime=runtime,
        manifest=None,
    )


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------


def _default_adapter_factory(
    router: "ServedModelRouter", entry: ServedModelEntry
) -> LocalDnaModelAdapter:
    """Default adapter factory.

    Prefers the entry's pre-built ``runtime`` (shared with ``entry.scorer`` so
    the predict and score paths resolve organism/metadata identically). Falls
    back to constructing a fresh runtime from the router's base runtime for
    hand-built entries that carry no runtime (e.g. tests).
    """
    if entry.runtime is not None:
        return LocalDnaModelAdapter(entry.runtime, scorer=entry.scorer)

    from alphagenome_pytorch.prediction import AlphaGenomePredictionRuntime

    runtime_kwargs: dict[str, Any] = dict(
        model=router.base_model,
        sequence_source=getattr(router.runtime, "sequence_source", None),
        metadata_catalog=entry.metadata_catalog
        if entry.metadata_catalog is not None
        else getattr(router.runtime, "metadata_catalog", None),
        track_names=entry.track_names,
        device=getattr(router.runtime, "device", None),
    )
    # Only forward a resolved default organism — mixed/unknown stays absent so
    # the runtime keeps its own default rather than silently forcing organism 0.
    if entry.default_organism is not None:
        runtime_kwargs["default_organism"] = entry.default_organism
    entry_runtime = AlphaGenomePredictionRuntime(**runtime_kwargs)
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
        self._native_heads = (
            dict(base_model.heads.items())
            if hasattr(base_model, "heads")
            else {}
        )
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
                row["base_model_weights_hash"] = e.manifest.base_model_weights_hash
                row["genome"] = e.manifest.genome
                row["organism"] = e.manifest.organism
                row["modalities"] = e.manifest.modalities
                row["biosample"] = e.manifest.biosample
                row["num_tracks"] = e.manifest.num_tracks
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

        Low-level: swaps the entry onto the shared base and returns an adapter,
        but the lock is released on return. Prefer :meth:`acquire`, which holds
        the lock across the model call so a concurrent request cannot re-swap
        the shared base mid-forward.
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

    @contextlib.contextmanager
    def acquire(self, model_id: str):
        """Hold the router lock across the adapter swap *and* the model call.

        Catalog mode serves one request at a time on the shared base model:
        the entry is swapped in and must stay resident until the caller finishes
        its prediction/scoring/attribution call. Usage::

            with router.acquire(model_id) as adapter:
                output = adapter.predict_sequence(...)

        The lock is a reentrant lock, so :meth:`select` re-acquiring inside is
        fine. Callers should keep only the model computation inside the ``with``
        block (serialization/streaming can happen after release, since outputs
        are materialized and no longer reference the swappable module tree).
        """
        with self._lock:
            yield self.select(model_id)

    # ---- swap mechanics --------------------------------------------------

    def _restore_native_heads_locked(self) -> None:
        _restore_head_modules(self.base_model, self._native_heads)

    def _detach_active_locked(self) -> None:
        if self._active_id is None:
            self._restore_native_heads_locked()
            return
        entry = self._entries[self._active_id]
        for att in entry.adapter_attachments:
            setattr(att.parent, att.attr_name, _wrapped_target(att.wrapper))
        self._restore_native_heads_locked()
        self._active_id = None

    def _attach_locked(self, entry: ServedModelEntry) -> None:
        self._restore_native_heads_locked()
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

"""Adapter bundle format: sidecar manifest beside an exported delta-weights file.

A bundle is a directory containing:

    my-bundle/
      adapter.safetensors          # produced by export_delta_weights (unchanged)
      alphagenome_adapter.json     # this module's manifest
      README.md                    # optional, generated model card
      metrics.json                 # optional, copied evaluation output

The manifest is **display/provenance only**. The loading truth lives in the
safetensors metadata (``transfer_config``, optional ``track_names`` and
``track_metadata``); the existing ``load_delta_weights`` /
``load_finetuned_model`` paths read those directly. The manifest's
``base_model_hash`` is cross-checked against the live base model at load time
so users get a clear error on mismatch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "alphagenome_adapter.json"
DEFAULT_ADAPTER_FILENAME = "adapter.safetensors"
README_FILENAME = "README.md"
METRICS_FILENAME = "metrics.json"

SCHEMA_VERSION = 1


class BundleError(Exception):
    """Raised on malformed bundles or compatibility failures."""


@dataclass
class Manifest:
    """Human/machine-readable bundle metadata.

    Only ``schema_version``, ``id``, and ``base_model_hash`` are required for
    loading; everything else is provenance/display.
    """

    id: str
    base_model_hash: str
    schema_version: int = SCHEMA_VERSION
    label: str | None = None
    base_model_id: str | None = None
    alphagenome_pytorch_version: str | None = None
    adapter_summary: dict[str, Any] = field(default_factory=dict)
    genome: str | None = None
    organism: str | None = None
    modality: str | None = None
    biosample: str | None = None
    heads: list[str] = field(default_factory=list)
    metrics_path: str | None = None
    license: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    adapter_filename: str = DEFAULT_ADAPTER_FILENAME

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        if not isinstance(data, dict):
            raise BundleError(
                f"Manifest must be a JSON object, got {type(data).__name__}"
            )
        schema = data.get("schema_version")
        if schema is None:
            raise BundleError("Manifest missing required field: schema_version")
        if schema > SCHEMA_VERSION:
            raise BundleError(
                f"Manifest schema_version {schema} is newer than this "
                f"alphagenome-pytorch supports ({SCHEMA_VERSION}). Upgrade "
                "alphagenome-pytorch to load this bundle."
            )
        for required in ("id", "base_model_hash"):
            if not data.get(required):
                raise BundleError(f"Manifest missing required field: {required}")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)

    @classmethod
    def load(cls, path: Path | str) -> "Manifest":
        p = Path(path)
        if p.is_dir():
            p = p / MANIFEST_FILENAME
        if not p.is_file():
            raise BundleError(f"Manifest not found: {p}")
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            raise BundleError(f"Manifest at {p} is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    def dump(self, path: Path | str) -> None:
        p = Path(path)
        if p.is_dir():
            p = p / MANIFEST_FILENAME
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


@dataclass
class BundlePaths:
    """Resolved on-disk locations within a bundle directory."""

    bundle_dir: Path
    manifest: Path
    adapter_safetensors: Path
    readme: Path | None
    metrics: Path | None

    @classmethod
    def resolve(cls, bundle_dir: Path | str) -> "BundlePaths":
        d = Path(bundle_dir)
        if not d.is_dir():
            raise BundleError(f"Bundle directory not found: {d}")
        manifest_path = d / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise BundleError(
                f"Bundle at {d} is missing {MANIFEST_FILENAME}"
            )
        manifest = Manifest.load(manifest_path)
        adapter_path = d / manifest.adapter_filename
        if not adapter_path.is_file():
            raise BundleError(
                f"Bundle at {d} declares adapter_filename={manifest.adapter_filename!r} "
                f"but {adapter_path.name} is missing"
            )
        readme = d / README_FILENAME
        metrics = d / METRICS_FILENAME
        return cls(
            bundle_dir=d,
            manifest=manifest_path,
            adapter_safetensors=adapter_path,
            readme=readme if readme.is_file() else None,
            metrics=metrics if metrics.is_file() else None,
        )


@dataclass
class ValidationReport:
    """Outcome of ``validate_bundle``."""

    bundle_dir: Path
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    manifest: Manifest | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_dir": str(self.bundle_dir),
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "manifest": self.manifest.to_dict() if self.manifest else None,
        }


def _read_safetensors_transfer_config(adapter_path: Path) -> dict[str, Any]:
    from alphagenome_pytorch.extensions.finetuning.checkpointing import (
        _read_delta_export_header,
    )

    header = _read_delta_export_header(adapter_path)
    return header.get("transfer_config", {})


def validate_bundle(
    bundle_dir: Path | str,
    base_model: Any | None = None,
) -> ValidationReport:
    """Validate a bundle directory.

    Checks:
    - Manifest schema and required fields.
    - Adapter safetensors file exists and embeds a ``transfer_config``.
    - If ``base_model`` is provided, ``compute_base_model_hash(base_model)``
      matches the manifest's ``base_model_hash``.
    - The manifest's ``adapter_summary.kind`` (if set) matches the
      ``transfer_config`` mode.

    Returns a ``ValidationReport``; never raises for normal validation errors
    (only for I/O issues that prevent reading the bundle at all).
    """
    bundle_dir = Path(bundle_dir)
    errors: list[str] = []
    warnings: list[str] = []
    manifest: Manifest | None = None

    try:
        paths = BundlePaths.resolve(bundle_dir)
        manifest = Manifest.load(paths.manifest)
    except BundleError as exc:
        return ValidationReport(
            bundle_dir=bundle_dir, ok=False, errors=[str(exc)]
        )

    try:
        transfer_config = _read_safetensors_transfer_config(paths.adapter_safetensors)
    except Exception as exc:
        errors.append(
            f"Could not read transfer_config from {paths.adapter_safetensors.name}: {exc}"
        )
        transfer_config = {}

    declared_kind = manifest.adapter_summary.get("kind") if manifest.adapter_summary else None
    if declared_kind and transfer_config:
        actual_mode = transfer_config.get("mode")
        actual_modes = actual_mode if isinstance(actual_mode, list) else [actual_mode]
        actual_modes = [m for m in actual_modes if m]
        if declared_kind not in actual_modes:
            warnings.append(
                f"Manifest adapter_summary.kind={declared_kind!r} is not present "
                f"in transfer_config.mode={actual_mode!r}"
            )

    if base_model is not None:
        try:
            from alphagenome_pytorch.extensions.finetuning.checkpointing import (
                compute_base_model_hash,
            )
            actual_hash = compute_base_model_hash(base_model)
        except Exception as exc:
            errors.append(f"Could not compute base_model_hash: {exc}")
        else:
            if actual_hash != manifest.base_model_hash:
                errors.append(
                    f"base_model_hash mismatch: manifest declares "
                    f"{manifest.base_model_hash!r} but supplied base model "
                    f"hashes to {actual_hash!r}"
                )

    return ValidationReport(
        bundle_dir=bundle_dir,
        ok=not errors,
        errors=errors,
        warnings=warnings,
        manifest=manifest,
    )


_MODEL_CARD_TEMPLATE = """\
---
library_name: alphagenome-pytorch
{base_model_block}license: {license}
tags:
- alphagenome
- adapter
{tag_block}---

# {label}

Adapter bundle exported by `agt adapters export`.

| Field | Value |
| --- | --- |
| Bundle id | `{id}` |
| Adapter | `{adapter_kind}` |
| Genome | {genome} |
| Organism | {organism} |
| Modality | {modality} |
| Biosample | {biosample} |
| Base model | `{base_model_id}` |
| Base model hash | `{base_model_hash}` |
| alphagenome-pytorch version | `{ag_version}` |

## Loading

```bash
agt serve --weights base.safetensors --checkpoint ./{adapter_filename}
```

The bundle's `alphagenome_adapter.json` describes the adapter for display and
provenance. The actual loading goes through `load_finetuned_model`, which
reads the embedded `transfer_config` from the safetensors metadata.
"""


def render_model_card(manifest: Manifest) -> str:
    """Render a Hugging Face model card from a manifest."""
    base_model_block = (
        f"base_model: {manifest.base_model_id}\nbase_model_relation: adapter\n"
        if manifest.base_model_id
        else ""
    )
    tags = []
    if manifest.modality:
        tags.append(manifest.modality)
    if manifest.organism:
        tags.append(manifest.organism)
    if manifest.genome:
        tags.append(manifest.genome)
    tag_block = "".join(f"- {t}\n" for t in tags)
    return _MODEL_CARD_TEMPLATE.format(
        base_model_block=base_model_block,
        license=manifest.license or "unknown",
        tag_block=tag_block,
        label=manifest.label or manifest.id,
        id=manifest.id,
        adapter_kind=manifest.adapter_summary.get("kind", "unknown"),
        genome=manifest.genome or "—",
        organism=manifest.organism or "—",
        modality=manifest.modality or "—",
        biosample=manifest.biosample or "—",
        base_model_id=manifest.base_model_id or "—",
        base_model_hash=manifest.base_model_hash,
        ag_version=manifest.alphagenome_pytorch_version or "—",
        adapter_filename=manifest.adapter_filename,
    )

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
``base_model_hash`` is cross-checked against the live base model at load time,
and newer bundles additionally verify ``base_model_weights_hash`` against the
base weights file so a compatible-but-wrong fold cannot be used silently.
"""

from __future__ import annotations

import json
import logging
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


def adapter_summary_kinds(adapter_summary: dict[str, Any] | None) -> list[str]:
    """Return the adapter kinds from a summary, preferring the ``kinds`` list.

    Falls back to the legacy scalar ``kind`` for bundles exported before the
    ``kinds`` list existed, so already-published bundles keep validating and
    displaying. Returns an empty list when neither field is present.
    """
    if not adapter_summary:
        return []
    kinds = adapter_summary.get("kinds")
    if isinstance(kinds, list):
        return [k for k in kinds if k]
    kind = adapter_summary.get("kind")
    return [kind] if kind else []


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
    modalities: list[str] = field(default_factory=list)
    biosample: str | None = None
    heads: list[str] = field(default_factory=list)
    num_tracks: int | None = None  # total tracks/tasks across all heads
    metrics_path: str | None = None
    license: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    adapter_filename: str = DEFAULT_ADAPTER_FILENAME
    # Appended to preserve the positional constructor order of schema v1.
    base_model_weights_hash: str | None = None
    base_model_variant: str | None = None

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


def short_base_model_hash(value: str | None) -> str:
    """Return a 16-character digest shorthand for human-facing output."""
    if not value:
        return "—"
    return value.rsplit(":", 1)[-1][:16]


short_base_model_weights_hash = short_base_model_hash


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
        adapter_filename = Path(manifest.adapter_filename).name
        adapter_path = d / adapter_filename
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
    base_weights_path: Path | str | None = None,
) -> ValidationReport:
    """Validate a bundle directory.

    Checks:
    - Manifest schema and required fields.
    - Adapter safetensors file exists and embeds a ``transfer_config``.
    - If ``base_model`` is provided, ``compute_base_model_hash(base_model)``
      matches the manifest's ``base_model_hash``.
    - If ``base_weights_path`` is provided, its canonical trunk tensor hash
      matches ``base_model_weights_hash`` when the bundle records one.
    - The manifest's ``adapter_summary`` kinds (if set) are present in the
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

    declared_kinds = adapter_summary_kinds(manifest.adapter_summary)
    if declared_kinds and transfer_config:
        actual_mode = transfer_config.get("mode")
        actual_modes = actual_mode if isinstance(actual_mode, list) else [actual_mode]
        actual_modes = [m for m in actual_modes if m]
        missing_kinds = [k for k in declared_kinds if k not in actual_modes]
        if missing_kinds:
            warnings.append(
                f"Manifest adapter_summary kinds {missing_kinds} are not present "
                f"in transfer_config.mode={actual_mode!r}"
            )

    if base_model is not None:
        try:
            from alphagenome_pytorch.extensions.finetuning.checkpointing import (
                base_model_structure_hashes_match,
                compute_base_model_hash,
            )
            actual_hash = compute_base_model_hash(base_model)
        except Exception as exc:
            errors.append(f"Could not compute base_model_hash: {exc}")
        else:
            if not base_model_structure_hashes_match(
                actual_hash, manifest.base_model_hash
            ):
                errors.append(
                    f"base_model_hash mismatch: manifest declares "
                    f"{manifest.base_model_hash!r} but supplied base model "
                    f"hashes to {actual_hash!r}"
                )

    if base_weights_path is not None:
        if manifest.base_model_weights_hash is None:
            warnings.append(
                "Manifest has no base_model_weights_hash; exact base-weight "
                "identity was not verified (legacy bundle)."
            )
        else:
            try:
                from alphagenome_pytorch.extensions.finetuning.checkpointing import (
                    compute_base_model_weights_hash_from_file,
                )
                actual_weights_hash = compute_base_model_weights_hash_from_file(
                    base_weights_path
                )
            except Exception as exc:
                errors.append(f"Could not compute base_model_weights_hash: {exc}")
            else:
                if actual_weights_hash != manifest.base_model_weights_hash:
                    errors.append(
                        "base_model_weights_hash mismatch: manifest declares "
                        f"{manifest.base_model_weights_hash!r} but supplied base "
                        f"weights hash to {actual_weights_hash!r}"
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
| Modalities | {modalities_display} |
| Heads | {n_heads} |
| Tracks | {n_tracks} |
| Biosample | {biosample} |
| Base model | `{base_model_id}` |
| Base model variant | `{base_model_variant}` |
| Base model structure hash | `{base_model_hash}` |
| Base model weights hash | `{base_model_weights_hash}` |
| alphagenome-pytorch version | `{ag_version}` |

## Usage

Install the CLI (`pip install 'alphagenome-pytorch[hf]'`), then fetch this
bundle and run it against a base model.

Download the bundle (this prints its local path) and point `FASTA` at the
reference genome your loci are relative to{genome_hint} — both `agt predict` and
`agt serve` require it:

```bash
BUNDLE=$(agt adapters pull hf://<org>/<repo>)
FASTA=/path/to/reference.fa
```

{predict_intro}

```bash
agt predict --model base.safetensors --checkpoint "$BUNDLE/{adapter_filename}" \\
    --fasta "$FASTA" \\
    --head {head} --locus chr1:1000000-1131072 --output ./predictions
```

Or serve it behind the AlphaGenome API:

```bash
agt serve --weights base.safetensors --checkpoint "$BUNDLE" --fasta "$FASTA"
```

`agt serve` takes the bundle *directory* (`"$BUNDLE"`), not the inner
safetensors file, so it can read `alphagenome_adapter.json` and verify the
manifest's base structure and exact weights hashes against `--weights` before
serving. The manifest is
otherwise display/provenance only: loading reads the embedded `transfer_config`
from the safetensors metadata via `load_finetuned_model`. Use a base model whose
hashes match those above.
"""


def render_model_card(manifest: Manifest) -> str:
    """Render a Hugging Face model card from a manifest."""
    base_model_block = (
        f"base_model: {manifest.base_model_id}\nbase_model_relation: adapter\n"
        if manifest.base_model_id
        else ""
    )
    tags = list(manifest.modalities)
    if manifest.organism:
        tags.append(manifest.organism)
    if manifest.genome:
        tags.append(manifest.genome)
    tag_block = "".join(f"- {t}\n" for t in tags)

    n_heads = len(manifest.heads)
    n_tracks = manifest.num_tracks if manifest.num_tracks is not None else "—"
    modalities_display = ", ".join(manifest.modalities) if manifest.modalities else "—"
    # A concrete --head for the predict example: prefer a real head name from the
    # bundle, fall back to the first modality, then a placeholder.
    head = (
        manifest.heads[0] if manifest.heads
        else (manifest.modalities[0] if manifest.modalities else "<head>")
    )
    # A multi-modality fine-tune registers one head per modality. `predict` runs a
    # single head per call, so note the choice when there is more than one — the
    # full head list stays in ``alphagenome_adapter.json`` rather than the card.
    if n_heads > 1:
        predict_intro = (
            f"Run one-off predictions. This bundle exposes {n_heads} heads; "
            f"`predict` runs one per call, selected with `--head` (see the "
            f"`heads` list in `alphagenome_adapter.json`):"
        )
    else:
        predict_intro = "Run one-off predictions:"
    # Name the reference genome when the manifest records it, so a non-human
    # bundle doesn't leave the reader guessing (or assuming hg38). Kept as a hint
    # beside the neutral ``FASTA`` variable — ``genome`` is a build label like
    # "mm10", not a path, so it can't be dropped straight into ``--fasta``.
    genome_hint = f" (this bundle was trained on {manifest.genome})" if manifest.genome else ""
    return _MODEL_CARD_TEMPLATE.format(
        genome_hint=genome_hint,
        base_model_block=base_model_block,
        license=manifest.license or "unknown",
        tag_block=tag_block,
        label=manifest.label or manifest.id,
        id=manifest.id,
        adapter_kind="+".join(adapter_summary_kinds(manifest.adapter_summary)) or "unknown",
        genome=manifest.genome or "—",
        organism=manifest.organism or "—",
        modalities_display=modalities_display,
        n_heads=n_heads,
        n_tracks=n_tracks,
        biosample=manifest.biosample or "—",
        base_model_id=manifest.base_model_id or "—",
        base_model_variant=manifest.base_model_variant or "—",
        base_model_hash=short_base_model_hash(manifest.base_model_hash),
        base_model_weights_hash=short_base_model_weights_hash(
            manifest.base_model_weights_hash
        ),
        ag_version=manifest.alphagenome_pytorch_version or "—",
        adapter_filename=manifest.adapter_filename,
        head=head,
        predict_intro=predict_intro,
    )


# ---------------------------------------------------------------------------
# Checkpoint resolution
#
# These live here rather than in serving/cli.py so that agt predict can
# accept the same bundle directories and URIs as agt serve without importing
# the serving CLI (and its gRPC extras) into the inference path. This module
# depends only on the stdlib, and huggingface_hub is imported lazily inside
# resolve_bundle, so nothing new reaches predict.
# ---------------------------------------------------------------------------

def resolve_checkpoint_and_manifest(checkpoint: str):
    """Resolve a ``--checkpoint`` value to ``(weights_path, manifest)``.

    ``weights_path`` is the concrete file the loader consumes; ``manifest`` is
    the bundle's parsed :class:`~...bundle.Manifest` when the checkpoint is a
    bundle (a directory or URI carrying ``alphagenome_adapter.json``), otherwise
    ``None``. The manifest is surfaced — not discarded — so the caller can
    cross-check ``base_model_hash`` before serving, the same compatibility
    guarantee catalog mode enforces in ``build_adapter_entry``.

    Accepts:
    - a path to a `.delta.pth` checkpoint or `.safetensors` delta-weights export,
    - a local bundle directory (one with ``alphagenome_adapter.json``),
    - any URI parseable by :func:`parse_bundle_uri` (``local:``, ``file:``,
      ``hf://``).

    Bundle directories resolve to the bundle's adapter safetensors path. Any
    other input — including missing files — is returned unchanged (with a
    ``None`` manifest) so the downstream loader can surface its native error.
    """
    from pathlib import Path

    from alphagenome_pytorch.extensions.serving.uri import (
        parse_bundle_uri,
        resolve_bundle,
    )

    # Plain filesystem paths (no URI scheme): just check if it's a bundle dir.
    if "://" not in checkpoint and not checkpoint.startswith(
        ("local:", "file:", "hf:")
    ):
        p = Path(checkpoint)
        if p.is_dir() and (p / MANIFEST_FILENAME).is_file():
            paths = BundlePaths.resolve(p)
            return str(paths.adapter_safetensors), Manifest.load(paths.manifest)
        return checkpoint, None

    parsed = parse_bundle_uri(checkpoint)
    if parsed.is_local:
        local = Path(parsed.path)
        if local.is_file():
            return str(local), None
        if local.is_dir() and (local / MANIFEST_FILENAME).is_file():
            paths = resolve_bundle(parsed)
            return str(paths.adapter_safetensors), Manifest.load(paths.manifest)
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # Remote URI (hf://). Resolve to a local bundle and return its adapter file.
    paths = resolve_bundle(parsed)
    return str(paths.adapter_safetensors), Manifest.load(paths.manifest)


def resolve_checkpoint_arg(checkpoint: str) -> str:
    """Map a ``--checkpoint`` value to a concrete weights path.

    Thin wrapper over :func:`resolve_checkpoint_and_manifest` for callers that
    only need the weights path.
    """
    return resolve_checkpoint_and_manifest(checkpoint)[0]


def verify_bundle_base_hash(model, manifest) -> None:
    """Refuse to serve a bundle whose base model is incompatible.

    Mirrors the check ``build_adapter_entry`` runs in catalog mode: the bundle's
    manifest records the ``base_model_hash`` (trunk structure) it was trained
    against; if the base ``model`` we just loaded hashes to something else, the
    adapter/head weights were built for a different architecture and would load
    incorrectly. ``compute_base_model_hash`` is structural and invariant to the
    applied adapters/merge, so it is safe to call on the fully-loaded model.
    """
    from alphagenome_pytorch.extensions.finetuning.checkpointing import (
        base_model_structure_hashes_match,
        compute_base_model_hash,
    )

    actual_hash = compute_base_model_hash(model)
    if not base_model_structure_hashes_match(actual_hash, manifest.base_model_hash):
        raise ValueError(
            f"Bundle {manifest.id!r} declares base_model_hash="
            f"{manifest.base_model_hash!r} but the base model hashes to "
            f"{actual_hash!r}. Refusing to serve an adapter on an incompatible "
            f"base — check that --weights matches the model the bundle was "
            f"trained against, or rebuild the bundle."
        )


def verify_bundle_base_weights_hash(
    weights_path: str,
    manifest,
    *,
    actual_hash: str | None = None,
) -> str | None:
    """Verify the exact base checkpoint/fold recorded by a bundle.

    Legacy manifests without an exact hash remain loadable, but emit a warning
    because only structural compatibility can be checked.
    """
    expected_hash = manifest.base_model_weights_hash
    if expected_hash is None:
        LOGGER.warning(
            "Bundle %r has no base_model_weights_hash; exact base-weight "
            "identity cannot be verified (legacy bundle).",
            manifest.id,
        )
        return actual_hash

    if actual_hash is None:
        from alphagenome_pytorch.extensions.finetuning.checkpointing import (
            compute_base_model_weights_hash_from_file,
        )
        actual_hash = compute_base_model_weights_hash_from_file(weights_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Bundle {manifest.id!r} declares base_model_weights_hash="
            f"{expected_hash!r} but --weights hashes to {actual_hash!r}. "
            "Refusing to serve the adapter on a different base checkpoint/fold."
        )
    return actual_hash

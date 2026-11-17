"""Server runner for ``agt serve``.

The argparse wiring lives in ``alphagenome_pytorch.cli.serve``; this module
holds the heavy-import server-startup logic so it is only loaded once the
user actually invokes the ``serve`` subcommand.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy
from alphagenome_pytorch.named_outputs import TrackMetadataCatalog
from alphagenome_pytorch.prediction import AlphaGenomePredictionRuntime
from alphagenome_pytorch.variant_scoring.inference import VariantScoringModel

from .adapter import LocalDnaModelAdapter
from .grpc_service import serve_grpc
from .rest_service import serve_rest
from .scorer import VariantScorer

LOGGER = logging.getLogger(__name__)


def _resolve_bundled_metadata_paths() -> list[Path]:
    """Locate built-in track metadata parquets shipped with the package.

    Mirrors the discovery in ``TrackMetadataCatalog.load_builtin`` so that
    ``agt serve`` can populate ``/v1/output_metadata`` with no explicit
    ``--track-metadata`` flag. The bundled files are split per organism, so
    both are returned (when present); each is suitable for
    ``VariantScoringModel.load_all_metadata`` because it carries an
    ``organism`` column.
    """
    paths: list[Path] = []
    try:
        import importlib.resources as resources

        files = resources.files('alphagenome_pytorch.data')
        for org_name in ('human', 'mouse'):
            candidate = files.joinpath(f'track_metadata_{org_name}.parquet')
            if hasattr(candidate, 'is_file') and candidate.is_file():
                paths.append(Path(str(candidate)))
    except (ImportError, ModuleNotFoundError):
        pass

    if paths:
        return paths

    # Fallback for installs where importlib.resources can't surface the data
    # directory (e.g. some zip-style installs). cli.py lives at
    # src/alphagenome_pytorch/extensions/serving/cli.py, so parents[2] is the
    # package root.
    module_data_dir = Path(__file__).resolve().parents[2] / 'data'
    for org_name in ('human', 'mouse'):
        candidate = module_data_dir / f'track_metadata_{org_name}.parquet'
        if candidate.exists():
            paths.append(candidate)
    return paths


def _load_metadata_catalog(
    args: argparse.Namespace,
    *,
    include_bundled: bool,
) -> TrackMetadataCatalog | None:
    """Load optional track metadata for serving.

    Pretrained weights can safely fall back to bundled metadata. Fine-tuned
    checkpoints may have custom/replaced heads, so their construction path only
    uses metadata explicitly provided by the user and otherwise relies on
    checkpoint ``track_names``.
    """
    if args.track_metadata:
        metadata_catalog = TrackMetadataCatalog.from_file(args.track_metadata)
        LOGGER.info('Loaded track metadata from %s', args.track_metadata)
        return metadata_catalog

    if not include_bundled:
        return None

    bundled_paths = _resolve_bundled_metadata_paths()
    if bundled_paths:
        metadata_catalog = TrackMetadataCatalog.from_file(bundled_paths[0])
        for path in bundled_paths[1:]:
            extra = TrackMetadataCatalog.from_file(path)
            metadata_catalog._tracks_by_organism.update(extra._tracks_by_organism)
        LOGGER.info(
            'Loaded built-in track metadata: %s',
            ', '.join(p.name for p in bundled_paths),
        )
        return metadata_catalog

    LOGGER.warning(
        'No track metadata available; /v1/output_metadata will be '
        'empty. Pass --track-metadata or reinstall the package so the '
        'bundled parquets ship under alphagenome_pytorch/data/.'
    )
    return None


def _sync_metadata_catalog_to_scoring_model(
    scoring_model: VariantScoringModel,
    metadata_catalog: TrackMetadataCatalog | None,
) -> None:
    """Copy runtime/catalog metadata into ``VariantScoringModel`` compatibility storage."""
    if metadata_catalog is None:
        return

    from alphagenome_pytorch.variant_scoring.types import (
        OutputType as PTOutputType,
        TrackMetadata as PTTrackMetadata,
    )

    for org_idx in metadata_catalog.organisms:
        for output_name in metadata_catalog.outputs(organism=org_idx):
            tracks = metadata_catalog.get_tracks(output_name, organism=org_idx)
            try:
                pt_output = PTOutputType(output_name)
            except ValueError:
                continue
            legacy_tracks = [
                PTTrackMetadata(
                    track_index=t.track_index,
                    track_name=t.track_name,
                    track_strand=t.get('strand', t.get('track_strand', '.')),
                    output_type=pt_output,
                    ontology_curie=t.get('ontology_curie'),
                    gtex_tissue=t.get('gtex_tissue'),
                    assay_title=t.get('assay_title'),
                    biosample_name=t.get('biosample_name'),
                    biosample_type=t.get('biosample_type'),
                    transcription_factor=t.get('transcription_factor'),
                    histone_mark=t.get('histone_mark'),
                )
                for t in tracks
            ]
            scoring_model.set_track_metadata(
                pt_output, legacy_tracks, organism=org_idx,
            )


def _make_variant_scorer(
    *,
    runtime: AlphaGenomePredictionRuntime,
    model: torch.nn.Module,
    args: argparse.Namespace,
    metadata_catalog: TrackMetadataCatalog | None,
) -> VariantScorer:
    """Build the optional variant-scoring capability for any AlphaGenome model."""
    scoring_model = VariantScoringModel(
        model=model,
        fasta_path=args.fasta,
        gtf_path=args.gtf,
        polya_path=args.polya,
        device=args.device,
    )
    _sync_metadata_catalog_to_scoring_model(scoring_model, metadata_catalog)
    return VariantScorer(runtime, scoring_model)


def _resolve_finetuned_metadata_catalog(
    args: argparse.Namespace,
    meta: dict,
) -> TrackMetadataCatalog | None:
    """Pick the right metadata source for a fine-tuned checkpoint.

    Order of precedence:

    1. ``--track-metadata`` from the CLI (explicit user override). When the
       checkpoint also embeds metadata, log a warning so the user knows the
       embedded catalog is being ignored.
    2. ``track_metadata`` rows embedded in the fine-tuned checkpoint
       (``finetune.py --track-metadata`` or ``export_delta_weights(...,
       track_metadata=...)``).
    3. ``None`` — the runtime falls back to bare ``track_names`` and serves
       sparse ``TrackMetadata`` entries.
    """
    embedded_rows = meta.get('track_metadata')

    if args.track_metadata:
        if embedded_rows:
            LOGGER.warning(
                'Both --track-metadata and an embedded metadata catalog were '
                'provided; using --track-metadata=%s. Drop the flag to use '
                'the embedded catalog.',
                args.track_metadata,
            )
        # Delegate to the shared loader (logs 'Loaded track metadata from %s').
        # include_bundled=False: fine-tuned heads may be custom, so don't fall
        # back to bundled pretrained metadata.
        return _load_metadata_catalog(args, include_bundled=False)

    if embedded_rows:
        catalog = TrackMetadataCatalog.from_rows(embedded_rows)
        if catalog.is_empty():
            LOGGER.warning(
                'Fine-tuned checkpoint embedded an empty track-metadata '
                'catalog; serving sparse track names instead.'
            )
            return None
        LOGGER.info('Using track metadata embedded in the fine-tuned checkpoint.')
        return catalog

    return None


# Checkpoint/bundle resolution now lives in ``bundle`` so ``agt predict`` can
# reuse it without importing this module. Re-exported under the original private
# names to keep existing callers and tests working.
from alphagenome_pytorch.extensions.serving.bundle import (  # noqa: E402
    resolve_checkpoint_and_manifest as _resolve_checkpoint_and_manifest,
    resolve_checkpoint_arg as _resolve_checkpoint_arg,
    verify_bundle_base_hash as _verify_bundle_base_hash,
    verify_bundle_base_weights_hash as _verify_bundle_base_weights_hash,
)


def _finetuned_default_organism(meta: dict) -> int:
    """Default organism index for a fine-tuned model, from the resolved metadata.

    Consumes the ``default_organism_index`` the canonical loader already resolved
    (checkpoint provenance + embedded catalog) — this must not re-run resolution.
    A mixed checkpoint has no single default (``None``); since mixed-organism serving
    is not yet supported, that fails at server construction rather than silently
    defaulting to human.
    """
    default = meta.get("default_organism_index")
    if default is None:
        raise ValueError(
            "This checkpoint has no single default organism. "
            "Mixed-organism serving is not yet supported."
        )
    return default


def _build_checkpoint_adapter(args: argparse.Namespace) -> LocalDnaModelAdapter:
    """Construct a serving adapter from a fine-tuned checkpoint."""
    from alphagenome_pytorch.extensions.finetuning.checkpointing import (
        load_finetuned_model,
    )
    from alphagenome_pytorch.extensions.finetuning.transfer import (
        transfer_config_from_dict,
    )

    transfer_config = None
    if args.transfer_config:
        with open(args.transfer_config) as f:
            transfer_config = transfer_config_from_dict(json.load(f))

    checkpoint_path, manifest = _resolve_checkpoint_and_manifest(args.checkpoint)

    if manifest is not None:
        _verify_bundle_base_weights_hash(args.weights, manifest)

    model, meta = load_finetuned_model(
        checkpoint_path=checkpoint_path,
        pretrained_weights=args.weights,
        device=args.device,
        dtype_policy=DtypePolicy.default(),
        transfer_config=transfer_config,
        merge=not args.no_merge_adapters,
    )
    # Bundle serving must honour the same base-model compatibility guarantee as
    # catalog mode (documented in docs/serving/adapters.rst). The manifest holds
    # the only copy of base_model_hash — the delta safetensors does not embed it
    # — so verify here now that resolution no longer discards the manifest.
    if manifest is not None:
        _verify_bundle_base_hash(model, manifest)
    metadata_catalog = _resolve_finetuned_metadata_catalog(args, meta)
    runtime = AlphaGenomePredictionRuntime(
        model=model,
        fasta_path=args.fasta,
        metadata_catalog=metadata_catalog,
        track_names=meta.get('track_names'),
        device=args.device,
        default_organism=_finetuned_default_organism(meta),
    )
    scorer = _make_variant_scorer(
        runtime=runtime,
        model=model,
        args=args,
        metadata_catalog=metadata_catalog,
    )
    LOGGER.info(
        'Loaded fine-tuned checkpoint %s (resolved to %s); variant scoring '
        'routes enabled for heads supported by the checkpoint.',
        args.checkpoint, checkpoint_path,
    )
    return LocalDnaModelAdapter(runtime, scorer=scorer)


def _build_weights_adapter(args: argparse.Namespace) -> LocalDnaModelAdapter:
    """Construct a variant-scoring adapter from a pretrained weights file."""
    model = AlphaGenome(num_organisms=2)
    state_dict = torch.load(args.weights, map_location=args.device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)
    model.eval()

    # Load track metadata via the canonical catalog path.
    metadata_catalog = _load_metadata_catalog(args, include_bundled=True)

    # Construct the runtime directly — no VariantScoringModel bridge.
    runtime = AlphaGenomePredictionRuntime(
        model=model,
        fasta_path=args.fasta,
        metadata_catalog=metadata_catalog,
        device=args.device,
    )

    scorer = _make_variant_scorer(
        runtime=runtime,
        model=model,
        args=args,
        metadata_catalog=metadata_catalog,
    )
    return LocalDnaModelAdapter(runtime, scorer=scorer)


def _build_adapter(args: argparse.Namespace) -> LocalDnaModelAdapter:
    """Pick the right adapter construction path based on CLI args.

    * ``--checkpoint`` → fine-tuned adapter with a configured ``VariantScorer``
    * ``--weights``   → pretrained adapter with a configured ``VariantScorer``
    """
    if args.checkpoint:
        return _build_checkpoint_adapter(args)
    return _build_weights_adapter(args)


def _load_catalog_base_weights(model, weights_path: str):
    """Load all base weights for catalog serving in pth or safetensors format."""
    from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk

    return load_trunk(
        model,
        weights_path,
        exclude_heads=False,
        strict=False,
    )


def _build_catalog_router(args: argparse.Namespace):
    """Construct a :class:`ServedModelRouter` from ``--adapter-catalog``."""
    from alphagenome_pytorch.extensions.serving.bundle import (
        BundlePaths,
        Manifest,
    )
    from alphagenome_pytorch.extensions.finetuning.checkpointing import (
        _read_delta_export_header,
        resolve_finetuned_organism,
    )
    from alphagenome_pytorch.extensions.serving.router import (
        ServedModelRouter,
        build_adapter_entry,
        build_base_entry,
        load_catalog,
    )
    from alphagenome_pytorch.extensions.serving.uri import resolve_bundle

    catalog = load_catalog(args.adapter_catalog)

    # Compute the canonical numerical identity once for every catalog bundle.
    from alphagenome_pytorch.extensions.finetuning.checkpointing import (
        compute_base_model_weights_hash_from_file,
    )
    actual_base_weights_hash = compute_base_model_weights_hash_from_file(args.weights)

    # Build base model + shared runtime exactly as in singleton-base mode, but
    # without constructing a singleton scorer (each entry brings its own).
    model = AlphaGenome(num_organisms=2)
    model = _load_catalog_base_weights(model, args.weights)
    model.to(args.device)
    model.eval()

    base_metadata_catalog = _load_metadata_catalog(args, include_bundled=True)
    runtime = AlphaGenomePredictionRuntime(
        model=model,
        fasta_path=args.fasta,
        metadata_catalog=base_metadata_catalog,
        device=args.device,
    )

    entries = []
    if catalog.base.enabled:
        base_scorer = _make_variant_scorer(
            runtime=runtime, model=model, args=args,
            metadata_catalog=base_metadata_catalog,
        )
        entries.append(build_base_entry(
            base_model=model,
            id=catalog.base.id,
            label=catalog.base.label,
            metadata_catalog=base_metadata_catalog,
            scorer=base_scorer,
            runtime=runtime,
        ))

    for spec in catalog.adapters:
        bundle_paths: BundlePaths = resolve_bundle(spec.source)
        manifest = Manifest.load(bundle_paths.manifest)
        _verify_bundle_base_weights_hash(
            args.weights, manifest, actual_hash=actual_base_weights_hash
        )
        if spec.label and not manifest.label:
            manifest.label = spec.label
        # Override the manifest id with the catalog's spec id so users can
        # alias bundles in their catalog file without rebuilding them.
        manifest.id = spec.id

        # Each served adapter describes itself: read its embedded delta header so
        # the entry gets its OWN track metadata, track names, organism default,
        # and variant scorer — not the base model's. An explicit --track-metadata
        # still overrides (via _resolve_finetuned_metadata_catalog).
        header = _read_delta_export_header(bundle_paths.adapter_safetensors)
        entry_catalog = _resolve_finetuned_metadata_catalog(args, header)
        entry_track_names = header.get("track_names")
        organism_ctx = resolve_finetuned_organism(
            organism_indices=header.get("organism_indices"),
            checkpoint_organism=header.get("organism"),
            track_metadata=header.get("track_metadata"),
            num_organisms=model.num_organisms,
        )
        if organism_ctx.default_organism_index is None:
            raise SystemExit(
                f"agt serve: adapter {spec.id!r} was fine-tuned on multiple "
                f"organisms {list(organism_ctx.organism_indices or [])}; "
                "multi-organism serving is not supported. Serve a "
                "single-organism bundle instead."
            )
        # One runtime per entry, shared by its service adapter AND its scorer so
        # the predict and score paths resolve organism/metadata identically —
        # in particular an organism-omitted request defaults to this bundle's
        # organism (e.g. mouse), not human. Reuse the base runtime's sequence
        # source (no second FASTA open) but carry this bundle's own metadata,
        # track names, and default organism.
        entry_runtime = AlphaGenomePredictionRuntime(
            model=model,
            sequence_source=runtime.sequence_source,
            metadata_catalog=entry_catalog,
            track_names=entry_track_names,
            device=args.device,
            default_organism=organism_ctx.default_organism_index,
        )
        entry_scorer = _make_variant_scorer(
            runtime=entry_runtime,
            model=model,
            args=args,
            metadata_catalog=entry_catalog,
        )
        # NB: build_adapter_entry mutates `model` and detaches before returning.
        entries.append(build_adapter_entry(
            base_model=model,
            bundle_paths=bundle_paths,
            manifest=manifest,
            metadata_catalog=entry_catalog,
            track_names=entry_track_names,
            scorer=entry_scorer,
            default_organism=organism_ctx.default_organism_index,
            runtime=entry_runtime,
            base_model_weights_hash=actual_base_weights_hash,
        ))

    if not entries:
        raise ValueError(
            "Adapter catalog produced no entries. Add a 'base:' block or at "
            "least one adapter to the catalog file."
        )

    return ServedModelRouter(
        base_model=model, runtime=runtime, entries=entries,
    )


def run(args: argparse.Namespace) -> int:
    """Start the serving process based on parsed *args*."""
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(levelname)s %(name)s - %(message)s',
    )

    catalog_path = getattr(args, "adapter_catalog", None)
    if catalog_path and args.checkpoint:
        raise SystemExit(
            "agt serve: --adapter-catalog and --checkpoint are mutually exclusive."
        )

    if catalog_path:
        target = _build_catalog_router(args)
        LOGGER.info("Catalog mode: serving %d models", len(target.model_ids))
    else:
        target = _build_adapter(args)

    grpc_server = None
    if not args.disable_grpc:
        grpc_server = serve_grpc(target, host=args.host, port=args.grpc_port, wait=False)
        LOGGER.info('gRPC ready at %s:%d', args.host, args.grpc_port)

    rest_server = None
    if args.rest_port is not None:
        rest_server = serve_rest(target, host=args.host, port=args.rest_port, wait=False)
        LOGGER.info('REST ready at http://%s:%d', args.host, args.rest_port)

    if grpc_server is None and rest_server is None:
        raise SystemExit("agt serve: at least one transport must be enabled (gRPC or REST).")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        LOGGER.info('Shutting down local serving...')
        if grpc_server is not None:
            grpc_server.stop(grace=3.0)
        if rest_server is not None:
            rest_server.shutdown()
            rest_server.server_close()
    return 0

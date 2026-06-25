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

    model, meta = load_finetuned_model(
        checkpoint_path=args.checkpoint,
        pretrained_weights=args.weights,
        device=args.device,
        dtype_policy=DtypePolicy.default(),
        transfer_config=transfer_config,
        merge=not args.no_merge_adapters,
    )
    metadata_catalog = _load_metadata_catalog(args, include_bundled=False)
    runtime = AlphaGenomePredictionRuntime(
        model=model,
        fasta_path=args.fasta,
        metadata_catalog=metadata_catalog,
        track_names=meta.get('track_names'),
        device=args.device,
    )
    scorer = _make_variant_scorer(
        runtime=runtime,
        model=model,
        args=args,
        metadata_catalog=metadata_catalog,
    )
    LOGGER.info(
        'Loaded fine-tuned checkpoint %s; variant scoring routes enabled '
        'for heads supported by the checkpoint.',
        args.checkpoint,
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


def run(args: argparse.Namespace) -> int:
    """Start the serving process based on parsed *args*."""
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(levelname)s %(name)s - %(message)s',
    )

    adapter = _build_adapter(args)

    grpc_server = None
    if not args.disable_grpc:
        grpc_server = serve_grpc(adapter, host=args.host, port=args.grpc_port, wait=False)
        LOGGER.info('gRPC ready at %s:%d', args.host, args.grpc_port)

    rest_server = None
    if args.rest_port is not None:
        rest_server = serve_rest(adapter, host=args.host, port=args.rest_port, wait=False)
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

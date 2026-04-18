"""Server runner for ``agt serve``.

The argparse wiring lives in ``alphagenome_pytorch.cli.serve``; this module
holds the heavy-import server-startup logic so it is only loaded once the
user actually invokes the ``serve`` subcommand.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.variant_scoring.inference import VariantScoringModel

from .adapter import LocalDnaModelAdapter
from .grpc_service import serve_grpc
from .rest_service import serve_rest

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


def run(args: argparse.Namespace) -> int:
    """Start the serving process based on parsed *args*."""
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(levelname)s %(name)s - %(message)s',
    )

    model = AlphaGenome(num_organisms=2)
    state_dict = torch.load(args.weights, map_location=args.device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)
    model.eval()

    scoring_model = VariantScoringModel(
        model=model,
        fasta_path=args.fasta,
        gtf_path=args.gtf,
        polya_path=args.polya,
        device=args.device,
    )
    if args.track_metadata:
        scoring_model.load_all_metadata(args.track_metadata)
        LOGGER.info('Loaded track metadata from %s', args.track_metadata)
    else:
        bundled_paths = _resolve_bundled_metadata_paths()
        if bundled_paths:
            for path in bundled_paths:
                scoring_model.load_all_metadata(path)
            LOGGER.info(
                'Loaded built-in track metadata: %s',
                ', '.join(p.name for p in bundled_paths),
            )
        else:
            LOGGER.warning(
                'No track metadata available; /v1/output_metadata will be '
                'empty. Pass --track-metadata or reinstall the package so the '
                'bundled parquets ship under alphagenome_pytorch/data/.'
            )

    adapter = LocalDnaModelAdapter(scoring_model)

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

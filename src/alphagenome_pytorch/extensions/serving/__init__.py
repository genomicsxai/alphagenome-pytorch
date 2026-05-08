"""Serving extensions for local AlphaGenome-compatible inference/scoring."""

from __future__ import annotations

from .adapter import (
    DEFAULT_MAX_WORKERS,
    SEQUENCE_LENGTH_100KB,
    SEQUENCE_LENGTH_16KB,
    SEQUENCE_LENGTH_1MB,
    SEQUENCE_LENGTH_500KB,
    SUPPORTED_SEQUENCE_LENGTHS,
    LocalDnaModelAdapter,
)

__all__ = [
    'LocalDnaModelAdapter',
    'SUPPORTED_SEQUENCE_LENGTHS',
    'SEQUENCE_LENGTH_16KB',
    'SEQUENCE_LENGTH_100KB',
    'SEQUENCE_LENGTH_500KB',
    'SEQUENCE_LENGTH_1MB',
    'DEFAULT_MAX_WORKERS',
]

# Optional transport exports (depend on grpc and/or HTTP server runtime usage).
try:  # pragma: no cover - exercised only when grpc dependencies are present.
    from .grpc_service import LocalDnaModelService, serve_grpc

    __all__.extend(['LocalDnaModelService', 'serve_grpc'])
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - exercised when REST server module is imported.
    from .rest_service import serve_rest

    __all__.append('serve_rest')
except Exception:  # pragma: no cover
    pass

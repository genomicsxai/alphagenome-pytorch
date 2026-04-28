"""REST (HTTP+JSON) transport for local AlphaGenome serving."""

from __future__ import annotations

import enum
import json
import logging
import threading
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import pandas as pd

from alphagenome.data import genome
from alphagenome.data import junction_data as ag_junction_data
from alphagenome.data import track_data as ag_track_data
from alphagenome.models import dna_output

from alphagenome_pytorch.variant_scoring.types import OutputType as PTOutputType

from .adapter import LocalDnaModelAdapter, _OFFICIAL_TO_PT_OUTPUT, _normalize_output_type
from alphagenome_pytorch.extensions.attribution import UnsupportedMethodError

# NOTE: variant-scoring scorer classes (``CenterMaskScorer`` etc.) and
# ``GeneMaskMode`` / ``AggregationType`` are deliberately *not* imported at
# module load time. They live in ``alphagenome_pytorch.variant_scoring.scorers``
# which is heavyweight (gene-mask, polyadenylation, splice-junction machinery).
# The ``--checkpoint`` (fine-tuned) serving path doesn't need any of that, so
# this module loads them on demand via :py:func:`_get_scorer_builders`.
#
# Caveat: ``alphagenome_pytorch.variant_scoring.__init__`` currently re-exports
# the scorers eagerly, which means importing *any* ``variant_scoring`` submodule
# (including ``variant_scoring.types`` below, used by the adapter) still pulls
# the scorers into ``sys.modules`` today. Fully decoupling rest_service from
# the scoring stack requires making that package init lazy too — out of scope
# for this PR. The lazy import here is still worthwhile: it establishes the
# boundary at this layer and benefits automatically once the package init is
# refactored.

LOGGER = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    """Fallback encoder for objects ``json.dumps`` can't handle natively.

    Covers the types we know leak through DataFrame ``to_dict`` calls in
    serializers below: enum members (notably ``dna_output.OutputType`` from
    ``OutputMetadata.concatenate``), numpy scalars, and numpy arrays. Anything
    else still raises ``TypeError`` so genuinely unexpected types fail loudly.
    """
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NaT or (isinstance(value, float) and np.isnan(value)):
        return None
    raise TypeError(
        f'Object of type {type(value).__name__} is not JSON serializable'
    )


def _interval_from_payload(payload: dict[str, Any]) -> genome.Interval:
    return genome.Interval(
        chromosome=payload['chromosome'],
        start=int(payload['start']),
        end=int(payload['end']),
        strand=payload.get('strand', '.'),
    )


def _variant_from_payload(payload: dict[str, Any]) -> genome.Variant:
    return genome.Variant(
        chromosome=payload['chromosome'],
        position=int(payload['position']),
        reference_bases=payload['reference_bases'],
        alternate_bases=payload['alternate_bases'],
    )


# Variant scorer JSON schema (tagged union on "type"):
#
#   {"type": "center_mask", "requested_output": "DNASE", "width": 501,
#    "aggregation_type": "DIFF_SUM"}
#   {"type": "contact_map"}
#   {"type": "gene_mask_lfc", "requested_output": "RNA_SEQ", "mask_mode": "EXONS"}
#   {"type": "gene_mask_active", "requested_output": "DNASE", "mask_mode": "EXONS"}
#   {"type": "gene_mask_splicing", "requested_output": "SPLICE_SITES", "width": 501}
#   {"type": "polyadenylation", "min_pas_count": 2, "min_pas_coverage": 0.8}
#   {"type": "splice_junction", "filter_protein_coding": true}
#
# Enum fields accept member names (case-insensitive). An empty or omitted
# "variant_scorers" list triggers adapter fallback to recommended scorers.


def _scorer_pt_output(payload: dict[str, Any]) -> PTOutputType:
    value = payload.get('requested_output')
    if value is None:
        raise ValueError(
            f'Variant scorer "{payload.get("type")}" missing required field "requested_output".'
        )
    official = _normalize_output_type(value)
    pt = _OFFICIAL_TO_PT_OUTPUT.get(official)
    if pt is None:
        raise ValueError(f'Output type {official.name} is not supported for variant scorers.')
    return pt


def _scorer_enum(
    payload: dict[str, Any],
    field: str,
    enum_cls: type[enum.Enum],
    *,
    default: Any = ...,
) -> Any:
    value = payload.get(field)
    if value is None:
        if default is ...:
            raise ValueError(
                f'Variant scorer "{payload.get("type")}" missing required field "{field}".'
            )
        return default
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls[value.upper()]
        except KeyError:
            valid = ', '.join(m.name for m in enum_cls)
            raise ValueError(
                f'Unknown {enum_cls.__name__} value "{value}". Valid: {valid}.'
            ) from None
    raise ValueError(f'Field "{field}" must be a string naming a {enum_cls.__name__} member.')


_SCORER_BUILDERS_CACHE: dict[str, Callable[[dict[str, Any]], Any]] | None = None


def _get_scorer_builders() -> dict[str, Callable[[dict[str, Any]], Any]]:
    """Lazily import variant-scoring classes and return the JSON→scorer dispatch table.

    Defers ``alphagenome_pytorch.variant_scoring.scorers`` (and the
    gene-mask / polyadenylation / splice-junction sub-modules it transitively
    pulls in) until a scoring request actually arrives. The fine-tuned
    serving path (``--checkpoint``) never enters this code, so its process
    avoids loading the scoring stack entirely.
    """
    global _SCORER_BUILDERS_CACHE
    if _SCORER_BUILDERS_CACHE is not None:
        return _SCORER_BUILDERS_CACHE

    from alphagenome_pytorch.variant_scoring.scorers import (
        CenterMaskScorer as PTCenterMaskScorer,
        ContactMapScorer as PTContactMapScorer,
        GeneMaskActiveScorer as PTGeneMaskActiveScorer,
        GeneMaskLFCScorer as PTGeneMaskLFCScorer,
        GeneMaskSplicingScorer as PTGeneMaskSplicingScorer,
        PolyadenylationScorer as PTPolyadenylationScorer,
        SpliceJunctionScorer as PTSpliceJunctionScorer,
    )
    from alphagenome_pytorch.variant_scoring.scorers.gene_mask import GeneMaskMode
    from alphagenome_pytorch.variant_scoring.types import AggregationType as PTAggregationType

    def _center_mask_from_json(payload: dict[str, Any]):
        width = payload.get('width')
        return PTCenterMaskScorer(
            requested_output=_scorer_pt_output(payload),
            width=int(width) if width is not None else None,
            aggregation_type=_scorer_enum(payload, 'aggregation_type', PTAggregationType),
        )

    def _contact_map_from_json(payload: dict[str, Any]):
        del payload
        return PTContactMapScorer()

    def _gene_mask_lfc_from_json(payload: dict[str, Any]):
        return PTGeneMaskLFCScorer(
            requested_output=_scorer_pt_output(payload),
            mask_mode=_scorer_enum(payload, 'mask_mode', GeneMaskMode, default=GeneMaskMode.EXONS),
        )

    def _gene_mask_active_from_json(payload: dict[str, Any]):
        return PTGeneMaskActiveScorer(
            requested_output=_scorer_pt_output(payload),
            mask_mode=_scorer_enum(payload, 'mask_mode', GeneMaskMode, default=GeneMaskMode.EXONS),
        )

    def _gene_mask_splicing_from_json(payload: dict[str, Any]):
        width = payload.get('width')
        return PTGeneMaskSplicingScorer(
            requested_output=_scorer_pt_output(payload),
            width=int(width) if width is not None else None,
        )

    def _polyadenylation_from_json(payload: dict[str, Any]):
        return PTPolyadenylationScorer(
            min_pas_count=int(payload.get('min_pas_count', 2)),
            min_pas_coverage=float(payload.get('min_pas_coverage', 0.8)),
        )

    def _splice_junction_from_json(payload: dict[str, Any]):
        return PTSpliceJunctionScorer(
            filter_protein_coding=bool(payload.get('filter_protein_coding', True)),
        )

    _SCORER_BUILDERS_CACHE = {
        'center_mask': _center_mask_from_json,
        'contact_map': _contact_map_from_json,
        'gene_mask_lfc': _gene_mask_lfc_from_json,
        'gene_mask_active': _gene_mask_active_from_json,
        'gene_mask_splicing': _gene_mask_splicing_from_json,
        'polyadenylation': _polyadenylation_from_json,
        'splice_junction': _splice_junction_from_json,
    }
    return _SCORER_BUILDERS_CACHE


def _parse_variant_scorers(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError('"variant_scorers" must be a JSON list of scorer objects.')
    builders = _get_scorer_builders()
    scorers: list[Any] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                f'variant_scorers[{index}] must be a JSON object, got {type(item).__name__}.'
            )
        scorer_type = item.get('type')
        if not isinstance(scorer_type, str):
            raise ValueError(
                f'variant_scorers[{index}] missing required string field "type". '
                f'Supported types: {", ".join(sorted(builders))}.'
            )
        builder = builders.get(scorer_type)
        if builder is None:
            raise ValueError(
                f'Unknown variant scorer type "{scorer_type}" at variant_scorers[{index}]. '
                f'Supported: {", ".join(sorted(builders))}.'
            )
        scorers.append(builder(item))
    return scorers


def _serialize_interval(interval: genome.Interval | None) -> dict[str, Any] | None:
    if interval is None:
        return None
    return {
        'chromosome': interval.chromosome,
        'start': interval.start,
        'end': interval.end,
        'strand': interval.strand,
    }


def _serialize_variant(variant: genome.Variant | None) -> dict[str, Any] | None:
    if variant is None:
        return None
    return {
        'chromosome': variant.chromosome,
        'position': variant.position,
        'reference_bases': variant.reference_bases,
        'alternate_bases': variant.alternate_bases,
    }


def _serialize_track_data(data: ag_track_data.TrackData) -> dict[str, Any]:
    return {
        'values': np.asarray(data.values).tolist(),
        'metadata': data.metadata.to_dict(orient='records'),
        'resolution': data.resolution,
        'interval': _serialize_interval(data.interval),
    }


def _serialize_junction_data(data: ag_junction_data.JunctionData) -> dict[str, Any]:
    junctions = [
        {
            'chromosome': j.chromosome,
            'start': j.start,
            'end': j.end,
            'strand': j.strand,
        }
        for j in data.junctions
    ]
    return {
        'junctions': junctions,
        'values': np.asarray(data.values).tolist(),
        'metadata': data.metadata.to_dict(orient='records'),
        'interval': _serialize_interval(data.interval),
    }


def _serialize_output(output: dna_output.Output) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in ['atac', 'cage', 'dnase', 'rna_seq', 'chip_histone', 'chip_tf', 'splice_sites', 'splice_site_usage', 'contact_maps', 'procap']:
        value = getattr(output, field)
        if value is None:
            payload[field] = None
        else:
            payload[field] = _serialize_track_data(value)
    splice_junctions = output.splice_junctions
    payload['splice_junctions'] = (
        _serialize_junction_data(splice_junctions) if splice_junctions is not None else None
    )
    return payload


def _serialize_variant_output(output: dna_output.VariantOutput) -> dict[str, Any]:
    return {
        'reference': _serialize_output(output.reference),
        'alternate': _serialize_output(output.alternate),
    }


def _serialize_anndata(adata: Any) -> dict[str, Any]:
    obs = adata.obs.to_dict(orient='records') if hasattr(adata, 'obs') else []
    var = adata.var.to_dict(orient='records') if hasattr(adata, 'var') else []
    uns = getattr(adata, 'uns', {}) or {}
    uns_payload = {
        'interval': _serialize_interval(uns.get('interval')),
        'variant': _serialize_variant(uns.get('variant')),
        'variant_scorer': str(uns.get('variant_scorer')) if 'variant_scorer' in uns else None,
    }
    return {
        'X': np.asarray(adata.X).tolist(),
        'obs': obs,
        'var': var,
        'uns': uns_payload,
    }


def _serialize_output_metadata(metadata: dna_output.OutputMetadata) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for output_type in dna_output.OutputType:
        data = metadata.get(output_type)
        outputs[output_type.name] = None if data is None else data.to_dict(orient='records')
    concatenated = metadata.concatenate() if any(metadata.get(o) is not None for o in dna_output.OutputType) else pd.DataFrame()
    return {
        'outputs': outputs,
        'concatenated': concatenated.to_dict(orient='records'),
    }


def _nan_to_none(arr: np.ndarray) -> list:
    """Convert a numpy array to a nested list with NaN replaced by None.

    ``ndarray.tolist()`` maps NaN to ``float('nan')``, which ``json.dumps``
    serializes as the non-standard ``NaN`` token.  We want ``null`` instead,
    so we replace NaN values with ``None`` in a Python object array first.
    """
    if not np.issubdtype(arr.dtype, np.floating):
        return arr.tolist()
    mask = np.isnan(arr)
    if not mask.any():
        return arr.tolist()
    obj = arr.astype(object)
    obj[mask] = None
    return obj.tolist()


def _serialize_attribution(result: 'AttributionResult') -> dict[str, Any]:
    """Convert an :class:`AttributionResult` to a JSON-serializable dict.

    NaN values in the numpy arrays are mapped to ``null`` via
    :func:`_nan_to_none`.
    """
    payload: dict[str, Any] = {
        'method': result.method,
        'kind': result.kind,
        'bases': list(result.bases),
        'values': _nan_to_none(result.values),
        'sequence': result.sequence,
        'target_start': result.target_start,
        'target_end': result.target_end,
        'resolution': result.resolution,
        'track_indices': list(result.track_indices),
        'reduction': result.reduction,
        'metadata': result.metadata,
    }
    if result.raw_gradient is not None:
        payload['raw_gradient'] = _nan_to_none(result.raw_gradient)
    else:
        payload['raw_gradient'] = None
    return payload


class _ServingHandler(BaseHTTPRequestHandler):
    adapter: LocalDnaModelAdapter

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get('Content-Length', '0'))
        body = self.rfile.read(content_length) if content_length else b'{}'
        if not body:
            return {}
        return json.loads(body.decode('utf-8'))

    def _write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(payload, default=_json_default).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path.startswith('/v1/output_metadata'):
                organism = self._parse_query_value('organism')
                metadata = self.adapter.output_metadata(organism=organism)
                self._write_json({'metadata': _serialize_output_metadata(metadata)})
                return
            self._write_json({'error': 'Not found'}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # pragma: no cover - exercised in integration.
            self._write_json({'error': str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._read_json()
            path = self.path.split('?', 1)[0]

            if path == '/v1/predict_sequence':
                output = self.adapter.predict_sequence(
                    sequence=body['sequence'],
                    organism=body.get('organism', 'HOMO_SAPIENS'),
                    requested_outputs=[_normalize_output_type(v) for v in body.get('requested_outputs', [])],
                    ontology_terms=body.get('ontology_terms'),
                )
                self._write_json({'output': _serialize_output(output)})
                return

            if path == '/v1/predict_interval':
                output = self.adapter.predict_interval(
                    interval=_interval_from_payload(body['interval']),
                    organism=body.get('organism', 'HOMO_SAPIENS'),
                    requested_outputs=[_normalize_output_type(v) for v in body.get('requested_outputs', [])],
                    ontology_terms=body.get('ontology_terms'),
                )
                self._write_json({'output': _serialize_output(output)})
                return

            if path == '/v1/predict_variant':
                output = self.adapter.predict_variant(
                    interval=_interval_from_payload(body['interval']),
                    variant=_variant_from_payload(body['variant']),
                    organism=body.get('organism', 'HOMO_SAPIENS'),
                    requested_outputs=[_normalize_output_type(v) for v in body.get('requested_outputs', [])],
                    ontology_terms=body.get('ontology_terms'),
                )
                self._write_json({'output': _serialize_variant_output(output)})
                return

            if path == '/v1/score_variant':
                if not getattr(self.adapter, 'supports_variant_scoring', False):
                    self._write_json(
                        {'error': 'Variant scoring not available for this model.'},
                        status=HTTPStatus.NOT_IMPLEMENTED,
                    )
                    return
                scores = self.adapter.score_variant(
                    interval=_interval_from_payload(body['interval']),
                    variant=_variant_from_payload(body['variant']),
                    variant_scorers=_parse_variant_scorers(body.get('variant_scorers')),
                    organism=body.get('organism', 'HOMO_SAPIENS'),
                )
                self._write_json({'scores': [_serialize_anndata(s) for s in scores]})
                return

            if path == '/v1/score_variants':
                if not getattr(self.adapter, 'supports_variant_scoring', False):
                    self._write_json(
                        {'error': 'Variant scoring not available for this model.'},
                        status=HTTPStatus.NOT_IMPLEMENTED,
                    )
                    return
                intervals_payload = body['intervals']
                variants_payload = body['variants']
                if isinstance(intervals_payload, dict):
                    intervals: genome.Interval | list[genome.Interval] = _interval_from_payload(intervals_payload)
                else:
                    intervals = [_interval_from_payload(i) for i in intervals_payload]
                variants = [_variant_from_payload(v) for v in variants_payload]
                scores = self.adapter.score_variants(
                    intervals=intervals,
                    variants=variants,
                    variant_scorers=_parse_variant_scorers(body.get('variant_scorers')),
                    organism=body.get('organism', 'HOMO_SAPIENS'),
                    progress_bar=False,
                    max_workers=int(body.get('max_workers', 5)),
                )
                self._write_json(
                    {'scores': [[_serialize_anndata(s) for s in group] for group in scores]}
                )
                return

            if path == '/v1/score_ism_variants':
                if not getattr(self.adapter, 'supports_variant_scoring', False):
                    self._write_json(
                        {'error': 'Variant scoring not available for this model.'},
                        status=HTTPStatus.NOT_IMPLEMENTED,
                    )
                    return
                interval_variant_payload = body.get('interval_variant')
                scores = self.adapter.score_ism_variants(
                    interval=_interval_from_payload(body['interval']),
                    ism_interval=_interval_from_payload(body['ism_interval']),
                    variant_scorers=_parse_variant_scorers(body.get('variant_scorers')),
                    organism=body.get('organism', 'HOMO_SAPIENS'),
                    interval_variant=_variant_from_payload(interval_variant_payload)
                    if interval_variant_payload
                    else None,
                    progress_bar=False,
                    max_workers=int(body.get('max_workers', 5)),
                )
                self._write_json(
                    {'scores': [[_serialize_anndata(s) for s in group] for group in scores]}
                )
                return

            if path == '/v1/explain_interval':
                try:
                    result = self.adapter.explain_interval(
                        interval=_interval_from_payload(body['interval']),
                        target_interval=_interval_from_payload(body['target_interval']),
                        organism=body.get('organism', 'HOMO_SAPIENS'),
                        requested_output=body['requested_output'],
                        resolution=int(body.get('resolution', 128)),
                        track_indices=[int(i) for i in body['track_indices']],
                        method=body['method'],
                        reduction=body.get('reduction', 'sum'),
                        include_raw_gradient=bool(body.get('include_raw_gradient', False)),
                        strand_averaged=bool(body.get('strand_averaged', False)),
                        batch_size=int(body.get('batch_size', 8)),
                    )
                except UnsupportedMethodError as exc:
                    self._write_json({'error': str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._write_json({'attribution': _serialize_attribution(result)})
                return

            self._write_json({'error': 'Not found'}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # pragma: no cover - exercised in integration.
            self._write_json({'error': str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        LOGGER.info('REST %s - %s', self.address_string(), format % args)

    def _parse_query_value(self, key: str) -> str | None:
        _, _, query = self.path.partition('?')
        if not query:
            return None
        for item in query.split('&'):
            k, _, v = item.partition('=')
            if k == key:
                return v
        return None


def _make_handler(adapter: LocalDnaModelAdapter):
    class Handler(_ServingHandler):
        pass

    Handler.adapter = adapter
    return Handler


def serve_rest(
    adapter: LocalDnaModelAdapter,
    *,
    host: str = '127.0.0.1',
    port: int = 8080,
    wait: bool = True,
) -> ThreadingHTTPServer:
    """Start a local REST server with JSON endpoints."""
    server = ThreadingHTTPServer((host, port), _make_handler(adapter))
    bound_host, bound_port = server.server_address
    LOGGER.info('Local REST serving started at http://%s:%d', bound_host, bound_port)

    if wait:
        server.serve_forever()
    else:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
    return server

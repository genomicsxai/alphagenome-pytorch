"""REST (HTTP+JSON) transport for local AlphaGenome serving."""

from __future__ import annotations

import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import pandas as pd

from alphagenome.data import genome
from alphagenome.data import junction_data as ag_junction_data
from alphagenome.data import track_data as ag_track_data
from alphagenome.models import dna_output

from .adapter import LocalDnaModelAdapter, _normalize_output_type

LOGGER = logging.getLogger(__name__)


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


class _ServingHandler(BaseHTTPRequestHandler):
    adapter: LocalDnaModelAdapter

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get('Content-Length', '0'))
        body = self.rfile.read(content_length) if content_length else b'{}'
        if not body:
            return {}
        return json.loads(body.decode('utf-8'))

    def _write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(payload).encode('utf-8')
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
                scores = self.adapter.score_variant(
                    interval=_interval_from_payload(body['interval']),
                    variant=_variant_from_payload(body['variant']),
                    variant_scorers=body.get('variant_scorers', ()),
                    organism=body.get('organism', 'HOMO_SAPIENS'),
                )
                self._write_json({'scores': [_serialize_anndata(s) for s in scores]})
                return

            if path == '/v1/score_variants':
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
                    variant_scorers=body.get('variant_scorers', ()),
                    organism=body.get('organism', 'HOMO_SAPIENS'),
                    progress_bar=False,
                    max_workers=int(body.get('max_workers', 5)),
                )
                self._write_json(
                    {'scores': [[_serialize_anndata(s) for s in group] for group in scores]}
                )
                return

            if path == '/v1/score_ism_variants':
                interval_variant_payload = body.get('interval_variant')
                scores = self.adapter.score_ism_variants(
                    interval=_interval_from_payload(body['interval']),
                    ism_interval=_interval_from_payload(body['ism_interval']),
                    variant_scorers=body.get('variant_scorers', ()),
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
    LOGGER.info('Local REST serving started at http://%s:%d', host, port)

    if wait:
        server.serve_forever()
    else:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
    return server


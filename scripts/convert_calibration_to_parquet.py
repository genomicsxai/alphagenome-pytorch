#!/usr/bin/env python
"""Convert upstream variant-score calibration protobuf to a bundled parquet.

This reads the precomputed ``calibration_scores.pb`` 
(the fixed null distribution of variant scores over 348,126 common chr22 gnomAD v3 SNPs,
see original at gs:///alphagenome/data/hg38/calibration_scores.pb,
added in https://github.com/google-deepmind/alphagenome_research/commit/dad09dd)
and writes ``src/alphagenome_pytorch/data/variant_quantile_calibration_<organism>.parquet``,
which is shipped as package data and loaded at runtime 
by ``alphagenome_pytorch.variant_scoring.calibration.Calibration``.

Only the human (hg38) calibration currently exists upstream.

Protobuf bindings
-----------------
Parsing the ``.pb`` uses the generated ``calibration_scores_pb2`` bindings, which
depend on ``alphagenome``'s ``dna_model.proto``. 

Example:
    python scripts/convert_calibration_to_parquet.py \
        --pb calibration_scores.pb \
        --alphagenome-src /path/to/alphagenome/src \
        --upstream-src /path/to/alphagenome_research/src \
        --metadata src/alphagenome_pytorch/data/track_metadata_human.parquet \
        --output src/alphagenome_pytorch/data/variant_quantile_calibration_human.parquet

Output schema (one parquet, long format)
-----------------------------------------
Columns: ``scorer_key`` (upstream canonical scorer string), ``output_type`` (our
output-type value, e.g. ``"atac"``), ``row_type`` (``"quantiles"`` or
``"probabilities"``), ``track_index`` (0..T-1 for quantiles, -1 for probabilities;
track indices align to the first T tracks of the scorer's head), ``is_signed``,
``values`` (list<float32> of length Q).
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Strand enum (dna_model.proto) -> our metadata convention.
_STRAND = {0: ".", 1: "+", 2: "-", 3: "."}


def _load_pb2(alphagenome_src: Path, upstream_src: Path):
    """Compile and import calibration_scores_pb2 from upstream proto sources."""
    try:
        return importlib.import_module(
            "alphagenome_research.protos.calibration_scores_pb2"
        )
    except Exception:  # noqa: BLE001 - fall through to on-the-fly compilation
        pass

    out_dir = Path(tempfile.mkdtemp(prefix="ag_calib_pb2_"))
    protos = [
        upstream_src / "alphagenome_research/protos/calibration_scores.proto",
        alphagenome_src / "alphagenome/protos/dna_model.proto",
        alphagenome_src / "alphagenome/protos/tensor.proto",
    ]
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={upstream_src}",
        f"--proto_path={alphagenome_src}",
        f"--python_out={out_dir}",
        *[str(p) for p in protos],
    ]
    subprocess.run(cmd, check=True)
    sys.path.insert(0, str(out_dir))
    return importlib.import_module(
        "alphagenome_research.protos.calibration_scores_pb2"
    )


def _ontology_curie(ontology_term) -> str:
    enum = ontology_term.DESCRIPTOR.fields_by_name["ontology_type"].enum_type
    name = enum.values_by_number[ontology_term.ontology_type].name
    prefix = name.replace("ONTOLOGY_TYPE_", "")
    return f"{prefix}:{ontology_term.id:07d}" if prefix else ""


def _calib_track_keys(calibration) -> list[tuple[str, str]]:
    return [
        (m.name, _STRAND[m.strand])
        for m in calibration.tracks_metadata.metadata
    ]


def _detect_output_type(
    track_keys: list[tuple[str, str]],
    metadata: pd.DataFrame,
) -> str:
    """Find the output_type whose first len(track_keys) tracks match exactly."""
    T = len(track_keys)
    for output_type, group in metadata.groupby("output_type"):
        group = group.sort_values("track_index")
        our_keys = list(zip(group["track_name"], group["strand"]))
        if len(our_keys) >= T and our_keys[:T] == track_keys:
            return str(output_type)
    raise ValueError(
        f"No output_type prefix-matches the {T} calibration tracks "
        f"(first track: {track_keys[0] if track_keys else 'n/a'})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pb", required=True, type=Path, help="calibration_scores.pb")
    parser.add_argument("--metadata", required=True, type=Path,
                        help="track_metadata_<organism>.parquet (track alignment)")
    parser.add_argument("--output", required=True, type=Path, help="output parquet")
    parser.add_argument("--alphagenome-src", type=Path,
                        help="alphagenome src dir (for proto compilation)")
    parser.add_argument("--upstream-src", type=Path,
                        help="alphagenome_research src dir (for proto compilation)")
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--compression-level", type=int, default=19)
    args = parser.parse_args()

    cpb = _load_pb2(args.alphagenome_src, args.upstream_src)
    metadata = pd.read_parquet(args.metadata)

    calibration_scores = cpb.CalibrationScores.FromString(args.pb.read_bytes())
    rows = []
    for scorer_key in sorted(calibration_scores.scorer_to_calibration.keys()):
        cal = calibration_scores.scorer_to_calibration[scorer_key]
        probabilities = np.asarray(cal.quantile_probabilities, dtype=np.float32)
        Q = probabilities.shape[0]
        track_keys = _calib_track_keys(cal)
        T = len(track_keys)
        quantiles = np.asarray(cal.quantiles, dtype=np.float32).reshape(T, Q)
        output_type = _detect_output_type(track_keys, metadata)
        is_signed = bool(probabilities.min() < 0)

        rows.append({
            "scorer_key": scorer_key,
            "output_type": output_type,
            "row_type": "probabilities",
            "track_index": -1,
            "is_signed": is_signed,
            "values": probabilities,
        })
        for track_index in range(T):
            rows.append({
                "scorer_key": scorer_key,
                "output_type": output_type,
                "row_type": "quantiles",
                "track_index": track_index,
                "is_signed": is_signed,
                "values": quantiles[track_index],
            })
        print(f"  {scorer_key[:60]:60s} T={T:5d} Q={Q} -> {output_type}")

    df = pd.DataFrame(rows)
    df["track_index"] = df["track_index"].astype(np.int32)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        args.output,
        compression=args.compression,
        compression_level=args.compression_level,
    )
    size_mb = args.output.stat().st_size / 1e6
    print(f"\nWrote {len(df)} rows for {df.scorer_key.nunique()} scorers "
          f"-> {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

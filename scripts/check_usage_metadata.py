#!/usr/bin/env python3
"""Validate usage-guide metadata facts against the bundled parquet catalogs."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_GUIDE = ROOT / "docs" / "alphagenome-usage.md"
VENDORED_GUIDE = (
    ROOT
    / "plugins"
    / "alphagenome"
    / "skills"
    / "alphagenome-predictions"
    / "reference"
    / "usage.md"
)
DATA_DIR = ROOT / "src" / "alphagenome_pytorch" / "data"

MAIN_HEADS = (
    "atac",
    "dnase",
    "procap",
    "cage",
    "rna_seq",
    "chip_tf",
    "chip_histone",
    "contact_maps",
)


def _load_catalog(organism: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"track_metadata_{organism}.parquet")


def _real_tracks(frame: pd.DataFrame) -> pd.DataFrame:
    names = frame["track_name"].fillna("").astype(str)
    return frame.loc[~names.str.casefold().eq("padding")]


def _nonempty_values(frame: pd.DataFrame, column: str) -> set[str]:
    values = frame[column].dropna().astype(str).str.strip()
    return {value for value in values if value}


def _parse_head_table(text: str) -> dict[str, tuple[int, int]]:
    rows: dict[str, tuple[int, int]] = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) < 6 or not cells[1].startswith("`"):
            continue
        head = cells[1].strip("`")
        if head not in MAIN_HEADS:
            continue
        try:
            rows[head] = (int(cells[3]), int(cells[4]))
        except ValueError:
            continue
    return rows


def _parse_literal_set(text: str, field: str, next_field: str) -> set[str]:
    pattern = rf"- `{re.escape(field)}` \(complete set\):(.*?)- `{re.escape(next_field)}`"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return set()
    return set(re.findall(r"`([^`]+)`", match.group(1)))


def _expected_head_table(human: pd.DataFrame) -> dict[str, tuple[int, int]]:
    real = _real_tracks(human)
    return {
        head: (
            int((human["output_type"] == head).sum()),
            int((real["output_type"] == head).sum()),
        )
        for head in MAIN_HEADS
    }


def validate_guide(text: str, human: pd.DataFrame, mouse: pd.DataFrame) -> list[str]:
    """Return human-readable validation failures for one guide."""
    errors: list[str] = []
    human_real = _real_tracks(human)
    mouse_real = _real_tracks(mouse)

    actual_table = _parse_head_table(text)
    expected_table = _expected_head_table(human)
    if actual_table != expected_table:
        errors.append(
            "main-head table does not match human metadata: "
            f"expected {expected_table}, found {actual_table}"
        )

    for field, next_field in (
        ("biosample_type", "assay_title"),
        ("assay_title", "biosample_name"),
    ):
        documented = _parse_literal_set(text, field, next_field)
        expected = _nonempty_values(human_real, field)
        if documented != expected:
            errors.append(
                f"documented {field} values do not match human metadata: "
                f"expected {sorted(expected)}, found {sorted(documented)}"
            )

    human_biosamples = _nonempty_values(human_real, "biosample_name")
    mouse_biosamples = _nonempty_values(mouse_real, "biosample_name")
    k562_tracks = int((human_real["biosample_name"] == "K562").sum())
    gm12878_tracks = int((human_real["biosample_name"] == "GM12878").sum())
    biosample_summary = (
        f"`biosample_name`: {len(human_biosamples)} distinct cell types/tissues "
        f"(human; {len(mouse_biosamples)} for mouse),\n"
        f"  including `K562` ({k562_tracks} tracks) and `GM12878` "
        f"({gm12878_tracks} tracks)"
    )
    if biosample_summary not in text:
        errors.append(f"biosample summary is stale; expected: {biosample_summary!r}")

    mouse_procap = int((mouse_real["output_type"] == "procap").sum())
    if f"`procap` has **{mouse_procap}**\nreal mouse tracks" not in text:
        errors.append(f"mouse PRO-cap count is stale; expected {mouse_procap}")

    splice_sites = int((human["output_type"] == "splice_sites").sum())
    splice_usage_human = int(
        (human_real["output_type"] == "splice_site_usage").sum()
    )
    splice_usage_mouse = int(
        (mouse_real["output_type"] == "splice_site_usage").sum()
    )
    splice_junctions = int((human["output_type"] == "splice_junctions").sum())
    splice_summary = (
        f"`splice_sites` (raw {splice_sites}),\n"
        f"`splice_site_usage` ({splice_usage_human} human / {splice_usage_mouse} mouse), "
        f"and `splice_junctions` ({splice_junctions} tissues → "
        f"{2 * splice_junctions} stranded output tracks)"
    )
    if splice_summary not in text:
        errors.append(f"splice-head summary is stale; expected: {splice_summary!r}")

    return errors


def main() -> int:
    canonical_bytes = CANONICAL_GUIDE.read_bytes()
    vendored_bytes = VENDORED_GUIDE.read_bytes()
    canonical = canonical_bytes.decode("utf-8")
    failures: list[str] = []

    if canonical_bytes != vendored_bytes:
        failures.append(
            "vendored usage guide differs from the canonical guide; re-sync with:\n"
            f"  cp {CANONICAL_GUIDE.relative_to(ROOT)} "
            f"{VENDORED_GUIDE.relative_to(ROOT)}"
        )

    human = _load_catalog("human")
    mouse = _load_catalog("mouse")
    for failure in validate_guide(canonical, human, mouse):
        failures.append(f"{CANONICAL_GUIDE.relative_to(ROOT)}: {failure}")

    if failures:
        print("Usage-guide validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("Usage guide mirror and metadata facts are current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

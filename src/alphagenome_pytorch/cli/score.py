"""agt score — variant effect prediction.

Wires the CLI to ``VariantScoringModel.score_variant``. The user picks one
or more named scorers (``--scorer atac,dnase,...``) or the bundled
``recommended`` set; for each variant we build an interval centered on
the variant position and emit per-track scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alphagenome_pytorch.cli._deps import require_extra
from alphagenome_pytorch.cli._output import emit_json


# Public list of accepted --scorer names (kept in sync with _build_scorer_registry).
SCORER_NAMES: list[str] = [
    "atac",
    "dnase",
    "chip_tf",
    "chip_histone",
    "cage",
    "procap",
    "contact_maps",
    "rna_seq",
    "rna_seq_active",
    "splice_sites",
    "splice_site_usage",
    "splice_junctions",
    "polyadenylation",
]


def _build_scorer_registry() -> dict:
    """Map scorer name -> zero-arg factory returning a list of scorer instances.

    Imports are lazy so ``--help`` and arg parsing don't pay the cost.
    """
    from alphagenome_pytorch.variant_scoring import (
        AggregationType,
        CenterMaskScorer,
        ContactMapScorer,
        GeneMaskActiveScorer,
        GeneMaskLFCScorer,
        GeneMaskSplicingScorer,
        OutputType,
        PolyadenylationScorer,
        SpliceJunctionScorer,
    )

    diff = AggregationType.DIFF_LOG2_SUM
    return {
        "atac": lambda: [CenterMaskScorer(OutputType.ATAC, 501, diff)],
        "dnase": lambda: [CenterMaskScorer(OutputType.DNASE, 501, diff)],
        "chip_tf": lambda: [CenterMaskScorer(OutputType.CHIP_TF, 501, diff)],
        "chip_histone": lambda: [CenterMaskScorer(OutputType.CHIP_HISTONE, 2001, diff)],
        "cage": lambda: [CenterMaskScorer(OutputType.CAGE, 501, diff)],
        "procap": lambda: [CenterMaskScorer(OutputType.PROCAP, 501, diff)],
        "contact_maps": lambda: [ContactMapScorer()],
        "rna_seq": lambda: [GeneMaskLFCScorer(OutputType.RNA_SEQ)],
        "rna_seq_active": lambda: [GeneMaskActiveScorer(OutputType.RNA_SEQ)],
        "splice_sites": lambda: [GeneMaskSplicingScorer(OutputType.SPLICE_SITES, width=None)],
        "splice_site_usage": lambda: [GeneMaskSplicingScorer(OutputType.SPLICE_SITE_USAGE, width=None)],
        "splice_junctions": lambda: [SpliceJunctionScorer()],
        "polyadenylation": lambda: [PolyadenylationScorer()],
    }


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "score",
        help="Score the impact of genetic variants",
        description=(
            "Variant effect prediction. Use --scorer (comma-separated names "
            "or 'recommended') to select scorers; use --vcf for batch scoring."
        ),
    )

    p.add_argument("--model", required=True, help="Path to model weights (.pth)")
    p.add_argument("--fasta", required=True, help="Path to reference genome FASTA")

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--variant", type=str,
        help='Single variant in default format, e.g. "chr22:36201698:A>C"',
    )
    source.add_argument("--vcf", type=str, help="Path to VCF file (TAB-separated)")

    p.add_argument(
        "--scorer",
        type=str,
        default="recommended",
        help=(
            "Comma-separated scorer names, or 'recommended' (default). "
            "Available: " + ", ".join(SCORER_NAMES) + "."
        ),
    )
    p.add_argument(
        "--organism", choices=["human", "mouse"], default="human",
        help="Organism for scoring (default: human)",
    )
    p.add_argument(
        "--width", type=int, default=131072,
        help="Interval width (bp) centered on each variant (default: 131072)",
    )
    p.add_argument("--gtf", type=str, default=None,
                   help="GTF annotation file (required for gene-centric scorers)")
    p.add_argument("--polya", type=str, default=None,
                   help="GENCODE polyAs file (used by PolyadenylationScorer)")
    p.add_argument("--output", type=str, default=None,
                   help="Output TSV path (ignored when --json is set)")
    p.add_argument("--device", type=str, default="cuda", help="PyTorch device")


def parse_vcf(path: Path) -> list:
    """Parse a minimal VCF (CHROM, POS, ID, REF, ALT) into Variant objects."""
    from alphagenome_pytorch.variant_scoring import Variant

    variants: list = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split("\t")
            if len(parts) < 5:
                raise ValueError(
                    f"VCF parse error at {path}:{lineno}: "
                    f"expected ≥5 tab-separated columns, got {len(parts)}"
                )
            chrom, pos, _id, ref, alt = parts[:5]
            try:
                pos_int = int(pos)
            except ValueError as e:
                raise ValueError(
                    f"VCF parse error at {path}:{lineno}: POS '{pos}' is not an integer"
                ) from e
            variants.append(Variant(
                chromosome=chrom,
                position=pos_int,
                reference_bases=ref,
                alternate_bases=alt,
                name=_id if _id and _id != "." else "",
            ))
    return variants


def resolve_scorers(spec: str, organism: str) -> list:
    """Turn the --scorer string into a list of scorer instances."""
    from alphagenome_pytorch.variant_scoring import get_recommended_scorers

    names = [n.strip().lower() for n in spec.split(",") if n.strip()]
    if not names:
        raise ValueError("--scorer cannot be empty")

    if names == ["recommended"]:
        return get_recommended_scorers(organism)

    if "recommended" in names:
        raise ValueError(
            "'recommended' cannot be combined with other scorer names"
        )

    registry = _build_scorer_registry()
    scorers: list = []
    for name in names:
        if name not in registry:
            raise ValueError(
                f"Unknown scorer '{name}'. "
                f"Available: {', '.join(SCORER_NAMES)}, recommended"
            )
        scorers.extend(registry[name]())
    return scorers


def _flatten(x) -> list[float]:
    """Flatten arbitrarily nested numeric lists to a flat list of floats."""
    if not isinstance(x, list):
        return [float(x)]
    out: list[float] = []
    for el in x:
        out.extend(_flatten(el))
    return out


def score_to_record(score) -> dict:
    """Serialize a VariantScore into a JSON-friendly dict."""
    import torch

    raw = score.scores
    if torch.is_tensor(raw):
        raw = raw.float().cpu().tolist()
    return {
        "variant": str(score.variant),
        "interval": str(score.interval),
        "scorer": score.scorer_name,
        "output_type": score.output_type.value,
        "is_signed": bool(score.is_signed),
        "gene_id": score.gene_id,
        "gene_name": score.gene_name,
        "gene_type": score.gene_type,
        "gene_strand": score.gene_strand,
        "junction_start": score.junction_start,
        "junction_end": score.junction_end,
        "scores": raw,
    }


def _flatten_results(score_results: list) -> list:
    """score_variant returns list[VariantScore | list[VariantScore]] — flatten it."""
    flat: list = []
    for entry in score_results:
        if isinstance(entry, list):
            flat.extend(entry)
        else:
            flat.append(entry)
    return flat


def _write_tsv(records: list[dict], path: Path) -> None:
    """Write per-track scores as TSV (one row per variant × scorer × track)."""
    cols = [
        "variant", "interval", "scorer", "output_type",
        "gene_id", "gene_name", "track_index", "raw_score",
    ]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for rec in records:
            base = [
                rec["variant"], rec["interval"], rec["scorer"], rec["output_type"],
                rec["gene_id"] or "", rec["gene_name"] or "",
            ]
            for i, val in enumerate(_flatten(rec["scores"])):
                f.write("\t".join(base + [str(i), f"{val:.6g}"]) + "\n")


def run(args: argparse.Namespace) -> int:
    require_extra("scoring", "score")

    json_mode = getattr(args, "json_output", False)

    for label, path in [("Model", args.model), ("FASTA", args.fasta)]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} file not found: {path}")
    for label, path in [("GTF", args.gtf), ("polyA", args.polya)]:
        if path is not None and not Path(path).exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.variant_scoring import (
        Interval,
        Variant,
        VariantScoringModel,
    )

    scorers = resolve_scorers(args.scorer, args.organism)

    if args.variant:
        variants = [Variant.from_str(args.variant)]
    else:
        vcf_path = Path(args.vcf)
        if not vcf_path.exists():
            raise FileNotFoundError(f"VCF file not found: {vcf_path}")
        variants = parse_vcf(vcf_path)

    if not variants:
        raise ValueError("No variants to score")

    if not json_mode:
        print(f"Loading model from {args.model}...")
    model = AlphaGenome.from_pretrained(args.model, device=args.device)
    model.eval()

    scoring_model = VariantScoringModel(
        model,
        fasta_path=args.fasta,
        gtf_path=args.gtf,
        polya_path=args.polya,
        default_organism=args.organism,
    )

    if not json_mode:
        print(f"Scoring {len(variants)} variant(s) with {len(scorers)} scorer(s)...")

    records: list[dict] = []
    for variant in variants:
        interval = Interval.centered_on(
            variant.chromosome, variant.position, width=args.width,
        )
        results = scoring_model.score_variant(
            interval, variant, scorers=scorers, to_cpu=True,
        )
        for s in _flatten_results(results):
            records.append(score_to_record(s))

    if json_mode:
        emit_json({"variants": records})
    elif args.output:
        out_path = Path(args.output)
        _write_tsv(records, out_path)
        print(f"Wrote {len(records)} score row(s) to {out_path}")
    else:
        for rec in records:
            flat = _flatten(rec["scores"])
            preview = ", ".join(f"{v:.4g}" for v in flat[:5])
            suffix = ", ..." if len(flat) > 5 else ""
            gene = f" gene={rec['gene_name']}" if rec["gene_name"] else ""
            print(f"{rec['variant']} | {rec['scorer']} | {rec['output_type']}{gene}")
            print(f"  scores ({len(flat)} tracks): [{preview}{suffix}]")

    return 0

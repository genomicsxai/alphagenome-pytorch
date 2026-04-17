"""agt score — variant effect prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alphagenome_pytorch.cli._deps import require_extra
from alphagenome_pytorch.cli._output import emit_json, emit_text


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "score",
        help="Score the impact of genetic variants",
        description="Variant effect prediction — score the impact of genetic variants.",
    )

    p.add_argument("--model", required=True, help="Path to model weights (.pth)")
    p.add_argument("--fasta", required=True, help="Path to reference genome FASTA")

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--variant", type=str, help='Single variant (e.g. "chr22:36201698:A>C")')
    source.add_argument("--vcf", type=str, help="Path to VCF file")

    p.add_argument("--head", type=str, default=None,
                    help="Score only this head (default: all)")
    p.add_argument("--scorer", type=str, default=None,
                    help="Aggregation method (e.g. logfc_max, logfc_mean)")
    p.add_argument("--output", type=str, default=None,
                    help="Output TSV file for VCF scoring")
    p.add_argument("--device", type=str, default="cuda", help="PyTorch device")


def run(args: argparse.Namespace) -> int:
    require_extra("scoring", "score")

    json_mode = getattr(args, "json_output", False)

    for label, path in [("Model", args.model), ("FASTA", args.fasta)]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.variant_scoring.types import Variant
    from alphagenome_pytorch.variant_scoring.inference import VariantScoringModel

    if not json_mode:
        print(f"Loading model from {args.model}...")
    model = AlphaGenome.from_pretrained(args.model, device=args.device)
    model.eval()

    scoring_model = VariantScoringModel(model, fasta_path=args.fasta)

    variants: list[Variant] = []
    if args.variant:
        variants.append(Variant.from_str(args.variant))
    elif args.vcf:
        if not Path(args.vcf).exists():
            raise FileNotFoundError(f"VCF file not found: {args.vcf}")
        # Parse VCF
        with open(args.vcf) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    chrom, pos, _, ref, alt = parts[:5]
                    variants.append(Variant(chrom=chrom, pos=int(pos), ref=ref, alt=alt))

    if not json_mode:
        print(f"Scoring {len(variants)} variant(s)...")

    results = []
    for variant in variants:
        scores = scoring_model.score_variant(variant, heads=([args.head] if args.head else None))
        result = {
            "chrom": variant.chrom,
            "pos": variant.pos,
            "ref": variant.ref,
            "alt": variant.alt,
            "scores": scores,
        }
        results.append(result)

    if json_mode:
        emit_json({"variants": results})
    elif args.output:
        # Write TSV
        with open(args.output, "w") as f:
            f.write("chrom\tpos\tref\talt\tscores\n")
            for r in results:
                import json as json_mod
                f.write(f"{r['chrom']}\t{r['pos']}\t{r['ref']}\t{r['alt']}\t{json_mod.dumps(r['scores'])}\n")
        print(f"Wrote scores to {args.output}")
    else:
        for r in results:
            print(f"{r['chrom']}:{r['pos']}:{r['ref']}>{r['alt']}")
            for head_name, head_scores in r["scores"].items():
                print(f"  {head_name}: {head_scores}")

    return 0

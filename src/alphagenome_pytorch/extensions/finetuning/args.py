"""Command-line argument definition for AlphaGenome finetuning.

Split out from the training implementation in
:mod:`alphagenome_pytorch.extensions.finetuning.runner` so the flags can be
declared on a parser without pulling in the training code. ``agt finetune``
registers them on its subparser at parser-build time, while ``runner`` is
imported only once a run actually starts.

Defining the flags here, once, is what keeps ``agt finetune`` and
``python scripts/finetune.py`` accepting an identical set of options.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from alphagenome_pytorch.extensions.finetuning.modalities import MODALITY_CONFIGS


DEFAULTS = {
    # Data
    "sequence_length": 131072,
    "resolutions": "1",
    # Model
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_targets": "q_proj,v_proj",
    "locon_rank": 4,
    "locon_alpha": 1,
    "locon_targets": "",
    # Training
    "epochs": 10,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "lr": 1e-4,
    "weight_decay": 0.1,
    "warmup_steps": 500,
    "lr_schedule": "cosine",
    "positional_weight": 5.0,
    "count_weight": 1.0,
    "num_workers": 4,
    "max_grad_norm": 1.0,
    "num_segments": 8,
    "min_segment_size": None,
    # Logging
    "wandb_project": "alphagenome-finetune",
    "log_every": 50,
    "save_every": 1,
    # Output
    "output_dir": "finetuning_output",
}


# =============================================================================
# CLI
# =============================================================================


def _normalize_strand_pairs(
    raw: object,
    n_bigwigs: int,
    modality: str,
    parser: argparse.ArgumentParser,
) -> list[tuple[int, int]] | None:
    """Normalize a strand-pair spec into [(plus_idx, minus_idx), ...].

    Accepts three forms (indices are 0-based into that modality's bigwig list):
      - 'auto'            : pair consecutive bigwigs (0,1),(2,3),... (needs even count)
      - 'p,m;p,m;...'     : CLI string of explicit pairs
      - [[p, m], ...]     : config list-of-lists of explicit pairs

    Index validity (bounds, distinctness, no reuse) is enforced downstream by
    compute_track_means; this only handles syntax and the 'auto' expansion.
    """
    if raw is None:
        return None
    if raw == "auto":
        if n_bigwigs % 2 != 0:
            parser.error(
                f"strand_pairs 'auto' for '{modality}' needs an even number of "
                f"bigwigs; got {n_bigwigs}"
            )
        return [(i, i + 1) for i in range(0, n_bigwigs, 2)]
    if isinstance(raw, str):
        chunks = [c for c in (s.strip() for s in raw.split(";")) if c]
        try:
            pairs = [tuple(int(x) for x in c.split(",")) for c in chunks]
        except ValueError:
            parser.error(
                f"strand_pairs for '{modality}' must be 'plus,minus' pairs; "
                f"got {raw!r}"
            )
    elif isinstance(raw, (list, tuple)):
        try:
            pairs = [tuple(int(x) for x in pair) for pair in raw]
        except (TypeError, ValueError):
            parser.error(
                f"strand_pairs for '{modality}' must be a list of [plus, minus] "
                f"pairs; got {raw!r}"
            )
    else:
        parser.error(f"strand_pairs for '{modality}' has unsupported type: {type(raw)}")
    if not pairs:
        parser.error(
            f"strand_pairs for '{modality}' is empty; specify 'auto' or at least "
            f"one 'plus,minus' pair, or omit it entirely to keep per-track means"
        )
    for pair in pairs:
        if len(pair) != 2:
            parser.error(
                f"strand_pairs for '{modality}' must contain exactly two indices "
                f"per pair; got {pair!r}"
            )
    return pairs


def _parse_cli_strand_pairs(
    spec: str | None,
    modalities: list[str],
    parser: argparse.ArgumentParser,
) -> dict[str, str]:
    """Parse '--strand-pairs' into {modality: raw_pairs_string} (unnormalized)."""
    result: dict[str, str] = {}
    if not spec:
        return result
    for entry in spec.split():
        if ":" not in entry:
            parser.error(f"--strand-pairs entry must be 'modality:pairs'; got {entry!r}")
        modality, pairs_str = entry.split(":", 1)
        modality = modality.strip()
        if modality not in modalities:
            parser.error(f"Unknown modality in --strand-pairs: {modality!r}")
        result[modality] = pairs_str.strip()
    return result


def add_finetune_arguments(parser: argparse.ArgumentParser) -> None:
    """Add every finetuning flag to *parser*.

    Shared by the standalone parser (:func:`build_parser`) and the
    ``agt finetune`` subparser, so the two never drift apart.
    """
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file (CLI flags override config values)",
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["linear-probe", "lora", "locon", "lora+locon", "full", "encoder-only"],
        default="lora",
        help=(
            "Training mode: "
            "'linear-probe' (frozen backbone, train heads on full transformer embeddings), "
            "'lora' (LoRA adapters + heads), "
            "'locon' (Locon adapters on convolutional layers + heads), "
            "'lora+locon' (LoRA on transformer targets plus Locon on convolutional targets + heads), "
            "'full' (all parameters), "
            "'encoder-only' (frozen backbone, train heads on raw CNN encoder output at 128bp; "
            "supports short sequences such as MPRA; forces --resolutions 128)"
        ),
    )

    # Data arguments
    data = parser.add_argument_group("Data")
    data.add_argument("--genome", type=str, required=False, help="Reference genome FASTA")
    data.add_argument(
        "--bigwig",
        type=str,
        nargs="+",
        action="append",
        dest="bigwigs",
        help="BigWig signal file(s). Repeat --bigwig for each modality when using multi-modality.",
    )
    data.add_argument("--train-bed", type=str, required=False, help="Training positions BED")
    data.add_argument("--val-bed", type=str, required=False, help="Validation positions BED")
    data.add_argument(
        "--track-metadata",
        type=str,
        default=None,
        help=(
            "Optional parquet/CSV/TSV with rich track metadata "
            "(ontology_curie, biosample_name, assay_title, ...). "
            "Embedded into checkpoints and exported delta weights so "
            "served models populate /v1/output_metadata without "
            "re-supplying --track-metadata at serve time. The 'output_type' "
            "column must match the head name (= --modality)."
        ),
    )
    data.add_argument(
        "--organism",
        type=str,
        choices=["human", "mouse"],
        default=None,
        help=(
            "Organism for the fine-tuning tracks when they all share one "
            "(human=0, mouse=1). Applied as the default for --track-metadata "
            "rows that lack an 'organism' value. For mixed human+mouse tracks, "
            "omit this and set a per-track 'organism' column in the parquet "
            "instead (it is authoritative). If --organism is given but the "
            "parquet declares a different or mixed set of organisms, "
            "fine-tuning errors out. Defaults to human (organism 0)."
        ),
    )
    data.add_argument("--sequence-length", type=int, default=DEFAULTS["sequence_length"])
    data.add_argument(
        "--resolutions",
        type=str,
        default=DEFAULTS["resolutions"],
        help="Comma-separated output resolutions (e.g., '1' or '1,128')",
    )
    data.add_argument(
        "--cache-genome",
        action="store_true",
        help="Cache genome in memory (~12GB for hg38)",
    )
    data.add_argument(
        "--cache-signals",
        action="store_true",
        help="Cache BigWig signals in memory (parallel init)",
    )
    data.add_argument(
        "--max-io-workers",
        type=int,
        default=16,
        help="Max threads for parallel BigWig I/O (default: 16)",
    )
    data.add_argument(
        "--gtf",
        type=str,
        default=None,
        help=(
            "Path to a GTF annotation file (Gencode-compatible). Required when "
            "--gene-loss-weight > 0. The GTF is parsed via pyranges and the "
            "protein_coding gene rows are used to build per-window gene masks."
        ),
    )
    data.add_argument(
        "--track-strands",
        type=str,
        default=None,
        help=(
            "Per-track strand string for the rna_seq modality, one char per "
            "BigWig in order. Each char must be '+', '-', or '.'. "
            "Compact ('+-+-') or separated ('+,-,+,-' / '+ - + -') forms "
            "are both accepted; commas and whitespace are stripped. "
            "Required when --gene-loss-weight > 0. Can also be supplied via "
            "the YAML config under modalities.<head>.strand."
        ),
    )
    data.add_argument(
        "--strand-pairs",
        type=str,
        default=None,
        help=(
            "Average +/- strand track means so paired strands share a scaling "
            "factor (recommended for stranded RNA-seq/CAGE/PRO-cap). Format: "
            "space-separated 'modality:pairs', where pairs is either 'auto' "
            "(pair consecutive bigwigs: (0,1),(2,3),...) or semicolon-separated "
            "'plus,minus' index pairs. Examples: --strand-pairs 'rna_seq:auto' "
            "or --strand-pairs 'rna_seq:0,1;2,3 cage:0,1'. Indices are 0-based "
            "into that modality's --bigwig list. Overrides per-modality "
            "'strand_pairs' from --config."
        ),
    )

    # Gene LFC loss arguments (Decima-style cross-track loss for RNA-seq).
    gene_lfc = parser.add_argument_group("Gene LFC loss")
    gene_lfc.add_argument(
        "--gene-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Outer weight on the cross-track gene LFC loss for the rna_seq "
            "head (paper value: 0.1). Default 0.0 disables the term entirely "
            "and keeps loss values bit-identical to pre-B3.2 behavior. "
            "When > 0, requires --gtf, rna_seq in --modality, and "
            "--track-strands (or modalities.rna_seq.strand in YAML)."
        ),
    )
    gene_lfc.add_argument(
        "--gene-cross-track-weight",
        type=float,
        default=5.0,
        help=(
            "Inner multinomial weight inside the gene LFC term (paper "
            "default: 5.0). Only used when --gene-loss-weight > 0."
        ),
    )

    # Gene-expression validation metric for the rna_seq head.
    gene_eval = parser.add_argument_group("Gene expression eval")
    gene_eval.add_argument(
        "--gene-expr-eval",
        action="store_true",
        help=(
            "Report gene-expression correlations for the "
            "rna_seq head each validation epoch: log-transformed mean coverage "
            "over annotated exons, strand-matched, keeping genes with >=50%% of "
            "their exons in-window. Emits rna_seq_gene_log_expr_pearson_* keys. "
            "Requires an annotation with exon rows (--gene-expr-annotation or "
            "--gtf) and rna_seq strands (--track-strands / config)."
        ),
    )
    gene_eval.add_argument(
        "--gene-expr-annotation",
        type=str,
        default=None,
        help=(
            "Annotation for --gene-expr-eval: a parquet (fast, recommended) or "
            "GTF/GFF (slow, needs pyranges) that INCLUDES exon rows. If unset, "
            "falls back to --gtf. Unlike --gtf's gene-only table (gene LFC loss), "
            "this needs exon features."
        ),
    )

    # Model arguments
    model = parser.add_argument_group("Model")
    model.add_argument("--pretrained-weights", type=str, required=False, help="Pretrained weights .pth")
    model.add_argument(
        "--modality",
        type=str,
        action="append",
        dest="modalities",
        choices=list(MODALITY_CONFIGS.keys()),
        help="Assay modality type. Repeat --modality for each --bigwig group in multi-modality mode.",
    )
    model.add_argument(
        "--modality-weights",
        type=str,
        default=None,
        help="Optional per-modality loss weights, e.g. 'atac:1.0,rna_seq:0.5'",
    )
    model.add_argument("--lora-rank", type=int, default=DEFAULTS["lora_rank"], help="LoRA rank (0 to disable)")
    model.add_argument("--lora-alpha", type=int, default=DEFAULTS["lora_alpha"], help="LoRA alpha scaling")
    model.add_argument(
        "--lora-targets",
        type=str,
        default=DEFAULTS["lora_targets"],
        help="Comma-separated modules for LoRA",
    )
    model.add_argument("--locon-rank", type=int, default=DEFAULTS["locon_rank"], help="Locon rank (0 to disable)")
    model.add_argument("--locon-alpha", type=int, default=DEFAULTS["locon_alpha"], help="Locon alpha scaling")
    model.add_argument(
        "--locon-targets",
        type=str,
        default=DEFAULTS["locon_targets"],
        help=(
            "Comma-separated modules for Locon (required when Locon is enabled). Examples: "
            "'down_blocks.5' for Locon2, 'down_blocks.4,down_blocks.5' for Locon4."
        ),
    )
    model.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Model dtype",
    )
    model.add_argument(
        "--head-init-scheme",
        type=str,
        default="truncated_normal",
        choices=["truncated_normal", "uniform"],
        help="Head weight initialization",
    )
    model.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")

    # Training arguments
    train = parser.add_argument_group("Training")
    train.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    train.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    train.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULTS["gradient_accumulation_steps"],
        help="Accumulate gradients over N batches",
    )
    train.add_argument("--lr", type=float, default=DEFAULTS["lr"], help="Learning rate")
    train.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    train.add_argument("--warmup-steps", type=int, default=DEFAULTS["warmup_steps"])
    train.add_argument(
        "--lr-schedule",
        type=str,
        default=DEFAULTS["lr_schedule"],
        choices=["cosine", "constant"],
    )
    train.add_argument("--positional-weight", type=float, default=DEFAULTS["positional_weight"])
    train.add_argument("--count-weight", type=float, default=DEFAULTS["count_weight"])
    train.add_argument("--max-grad-norm", type=float, default=DEFAULTS["max_grad_norm"])
    train.add_argument("--num-segments", type=int, default=DEFAULTS["num_segments"])
    train.add_argument("--min-segment-size", type=int, default=DEFAULTS["min_segment_size"])
    train.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    train.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    train.add_argument("--track-means-samples", type=int, default=None, help="Samples for track means (default: all)")
    train.add_argument("--profile-batches", type=int, default=0, help="Profile first N batches")
    train.add_argument("--compile", action="store_true", help="Use torch.compile")
    train.add_argument("--seed", type=int, default=None, help="Random seed")

    # Distributed/Sequence Parallel arguments
    dist = parser.add_argument_group("Distributed")
    dist.add_argument(
        "--sequence-parallel",
        action="store_true",
        help="Enable sequence parallelism (split sequence across GPUs)",
    )
    dist.add_argument(
        "--overlap-highres",
        type=int,
        default=1024,
        help="Overlap for high-resolution (1bp) sequence splits. Low-resolution overlap is computed as overlap_highres // 128.",
    )

    # Logging arguments
    log = parser.add_argument_group("Logging")
    log.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    log.add_argument("--wandb-project", type=str, default=DEFAULTS["wandb_project"])
    log.add_argument("--wandb-entity", type=str, default=None)
    log.add_argument("--log-every", type=int, default=DEFAULTS["log_every"])

    # Output arguments
    out = parser.add_argument_group("Output")
    out.add_argument("--output-dir", type=str, default=DEFAULTS["output_dir"])
    out.add_argument("--run-name", type=str, default=None)
    out.add_argument("--save-every", type=int, default=DEFAULTS["save_every"])
    out.add_argument("--no-save-checkpoints", action="store_true", help="Skip saving model checkpoints (keeps logs/config)")

    # Resume arguments
    resume = parser.add_argument_group("Resume / Checkpointing")
    resume.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path or 'auto' to find latest",
    )
    resume.add_argument(
        "--save-delta",
        action="store_true",
        help="Save delta checkpoints (adapter + head weights only, much smaller) "
             "for both best-model and per-epoch saves, alongside full checkpoints.",
    )
    resume.add_argument(
        "--no-full-checkpoint",
        action="store_true",
        help="Skip writing full checkpoints (best_model.pth, checkpoint_epoch*.pth). "
             "Requires --save-delta so the run still produces loadable checkpoints.",
    )
    resume.add_argument(
        "--export-transfer-config",
        type=str,
        default=None,
        metavar="PATH",
        help="Export TransferConfig to JSON file at end of training. "
             "Useful for loading full checkpoints in predict scripts.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone finetuning parser."""
    parser = argparse.ArgumentParser(
        description="Unified AlphaGenome training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_finetune_arguments(parser)
    return parser


def postprocess_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    tokens: list[str],
) -> argparse.Namespace:
    """Validate *args*, merge any YAML config, and derive the training fields.

    *tokens* is the raw argument list; it is used to decide which flags were
    passed explicitly, since those take precedence over the YAML config.
    *parser* is used to report errors against the parser the user invoked.
    """
    if args.no_full_checkpoint and not args.save_delta:
        parser.error(
            "--no-full-checkpoint requires --save-delta (otherwise the run "
            "would produce no loadable checkpoints)."
        )
    if args.save_delta and args.mode == "full":
        parser.error(
            "--save-delta cannot be used with --mode full: delta checkpoints "
            "only store adapter/head/norm weights, so they would omit all "
            "fine-tuning updates to the trunk."
        )
    cli_flags = {
        token.split("=", 1)[0]
        for token in tokens
        if token.startswith("--")
    }

    def _load_yaml_config(path: str) -> dict[str, Any]:
        try:
            import yaml
        except ImportError:
            parser.error("YAML config support requires PyYAML (`pip install pyyaml`).")
        config_path = Path(path)
        if not config_path.exists():
            parser.error(f"Config file not found: {config_path}")
        with config_path.open() as f:
            data = yaml.safe_load(f)
        if data is None:
            return {}
        if not isinstance(data, dict):
            parser.error("YAML config root must be a mapping/dictionary")
        return data

    def _parse_resolutions_value(value: Any, context: str) -> tuple[int, ...]:
        if isinstance(value, int):
            parsed = (int(value),)
        elif isinstance(value, str):
            parsed = tuple(int(r.strip()) for r in value.split(",") if r.strip())
        elif isinstance(value, (list, tuple)):
            parsed = tuple(int(r) for r in value)
        else:
            parser.error(f"Invalid resolutions for {context}: {value!r}")
        if not parsed:
            parser.error(f"Empty resolutions for {context}")
        return parsed

    def _apply_config_scalar(attr: str, config: dict[str, Any], key: str | None = None) -> None:
        flag = f"--{attr.replace('_', '-')}"
        if flag in cli_flags:
            return
        config_key = key or attr
        if config_key in config and config[config_key] is not None:
            setattr(args, attr, config[config_key])

    def _parse_weight_overrides(raw: Any) -> dict[str, float]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return {str(k): float(v) for k, v in raw.items()}
        if isinstance(raw, str):
            out: dict[str, float] = {}
            for item in raw.split(","):
                item = item.strip()
                if not item:
                    continue
                if ":" not in item:
                    parser.error("Weights must be specified as modality:weight pairs")
                mod, weight = item.split(":", 1)
                out[mod.strip()] = float(weight.strip())
            return out
        parser.error("Task weights in config must be a dict or comma-separated string")

    config_data = _load_yaml_config(args.config) if args.config else {}
    args.config_data = config_data
    args.config_path = args.config

    # Scalar config values (CLI wins)
    for attr in (
        "mode",
        "genome",
        "train_bed",
        "val_bed",
        "sequence_length",
        "resolutions",
        "cache_genome",
        "cache_signals",
        "max_io_workers",
        "pretrained_weights",
        "lora_rank",
        "lora_alpha",
        "lora_targets",
        "locon_rank",
        "locon_alpha",
        "locon_targets",
        "dtype",
        "head_init_scheme",
        "gradient_checkpointing",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "lr",
        "weight_decay",
        "warmup_steps",
        "lr_schedule",
        "positional_weight",
        "count_weight",
        "max_grad_norm",
        "num_segments",
        "min_segment_size",
        "num_workers",
        "track_means_samples",
        "profile_batches",
        "compile",
        "seed",
        "wandb",
        "wandb_project",
        "wandb_entity",
        "log_every",
        "output_dir",
        "run_name",
        "save_every",
        "resume",
        "modality_weights",
        "track_metadata",
        "gtf",
        "track_strands",
        "gene_loss_weight",
        "gene_cross_track_weight",
        "gene_expr_annotation",
    ):
        _apply_config_scalar(attr, config_data)

    if "--gene-expr-eval" not in cli_flags and "gene_expr_eval" in config_data:
        args.gene_expr_eval = bool(config_data["gene_expr_eval"])

    # Boolean aliases / migration-friendly keys
    if "--no-amp" not in cli_flags:
        if "use_amp" in config_data:
            args.no_amp = not bool(config_data["use_amp"])
        elif "no_amp" in config_data:
            args.no_amp = bool(config_data["no_amp"])

    if "--cache-genome" not in cli_flags and "--cache-signals" not in cli_flags:
        if bool(config_data.get("no_cache", False)):
            args.cache_genome = False
            args.cache_signals = False

    args.global_resolutions = _parse_resolutions_value(args.resolutions, "global resolutions")
    args.resolutions = ",".join(str(r) for r in args.global_resolutions)

    raw_modalities = config_data.get("modalities", {}) or {}
    if not isinstance(raw_modalities, dict):
        parser.error("Config key 'modalities' must be a mapping of modality -> settings")

    modality_specs: dict[str, dict[str, Any]] = {}
    for modality, mod_cfg in raw_modalities.items():
        if modality not in MODALITY_CONFIGS:
            parser.error(f"Unknown modality in config: {modality}")
        if not isinstance(mod_cfg, dict):
            parser.error(f"modalities.{modality} must be a mapping")
        spec: dict[str, Any] = {}
        if "bigwig" in mod_cfg and mod_cfg["bigwig"] is not None:
            bigwigs = mod_cfg["bigwig"]
            if isinstance(bigwigs, str):
                bigwigs = [bigwigs]
            if not isinstance(bigwigs, list):
                parser.error(f"modalities.{modality}.bigwig must be a string or list")
            spec["bigwig"] = [str(p) for p in bigwigs]
        if "resolutions" in mod_cfg and mod_cfg["resolutions"] is not None:
            spec["resolutions"] = _parse_resolutions_value(
                mod_cfg["resolutions"], f"modalities.{modality}.resolutions"
            )
        if "task_weight" in mod_cfg and mod_cfg["task_weight"] is not None:
            spec["task_weight"] = float(mod_cfg["task_weight"])
        if "strand" in mod_cfg and mod_cfg["strand"] is not None:
            strand_val = mod_cfg["strand"]
            if isinstance(strand_val, list):
                strand_val = "".join(str(s) for s in strand_val)
            elif not isinstance(strand_val, str):
                parser.error(
                    f"modalities.{modality}.strand must be a string of "
                    f"+/-/. characters or a list, got {type(strand_val).__name__}"
                )
            spec["strand"] = strand_val
        if "strand_pairs" in mod_cfg and mod_cfg["strand_pairs"] is not None:
            spec["strand_pairs"] = mod_cfg["strand_pairs"]
        modality_specs[modality] = spec

    cli_modality_to_bigwigs: dict[str, list[str]] = {}
    if args.bigwigs is not None:
        if args.modalities is None:
            parser.error(
                "--modality is required when --bigwig is provided. "
                f"Pass one of: {sorted(MODALITY_CONFIGS.keys())}."
            )
        if len(args.modalities) != len(args.bigwigs):
            parser.error(
                f"Number of --modality ({len(args.modalities)}) must match number of --bigwig groups ({len(args.bigwigs)}). "
                "Each --modality should be followed by --bigwig FILES."
            )
        for modality, bigwigs in zip(args.modalities, args.bigwigs):
            if modality in cli_modality_to_bigwigs:
                parser.error(f"Duplicate modality: {modality}")
            cli_modality_to_bigwigs[modality] = bigwigs
    elif args.modalities is not None and "--modality" in cli_flags:
        parser.error("--modality requires matching --bigwig entries")

    for modality, bigwigs in cli_modality_to_bigwigs.items():
        merged = dict(modality_specs.get(modality, {}))
        merged["bigwig"] = list(bigwigs)
        modality_specs[modality] = merged

    args.modalities = list(modality_specs.keys())
    if not args.modalities:
        parser.error("--bigwig is required (or provide modalities in --config)")

    # Required scalar args after config merge
    for flag, value in (
        ("--genome", args.genome),
        ("--train-bed", args.train_bed),
        ("--val-bed", args.val_bed),
        ("--pretrained-weights", args.pretrained_weights),
    ):
        if not value:
            parser.error(f"{flag} is required (or provide it in --config)")

    cli_strand_pairs = _parse_cli_strand_pairs(args.strand_pairs, args.modalities, parser)

    args.modality_to_bigwigs = {}
    args.modality_resolutions = {}
    args.modality_weight_dict = {}
    args.modality_strands: dict[str, str] = {}
    args.modality_strand_pairs = {}
    for modality in args.modalities:
        spec = modality_specs.get(modality, {})
        if "bigwig" not in spec or not spec["bigwig"]:
            parser.error(f"No bigwig files specified for modality '{modality}'")
        args.modality_to_bigwigs[modality] = list(spec["bigwig"])
        args.modality_resolutions[modality] = spec.get("resolutions", args.global_resolutions)
        args.modality_weight_dict[modality] = float(spec.get("task_weight", 1.0))
        if spec.get("strand"):
            # str ('+-+-') or a YAML list (['+','-',...]); normalized below.
            args.modality_strands[modality] = spec["strand"]
        # CLI --strand-pairs overrides per-modality config 'strand_pairs'.
        raw_pairs = cli_strand_pairs.get(modality, spec.get("strand_pairs"))
        args.modality_strand_pairs[modality] = _normalize_strand_pairs(
            raw_pairs, len(args.modality_to_bigwigs[modality]), modality, parser
        )

    # `--track-strands` CLI flag overrides any YAML strand for rna_seq.
    if args.track_strands:
        if "rna_seq" not in args.modality_to_bigwigs:
            parser.error(
                "--track-strands is only meaningful with --modality rna_seq, "
                f"but modalities are: {sorted(args.modality_to_bigwigs)}"
            )
        args.modality_strands["rna_seq"] = args.track_strands

    # Normalize and validate strand specs. Accept a compact string ('+-+-.-'),
    # a comma/whitespace-separated string ('+,-,+,-,.,-' or '+ - + - . -'), or a
    # YAML list of chars (['+','-','+','-']). After coercing to a bare string,
    # exactly one char per bigwig, each in {+, -, .}.
    def _normalize_strand_string(s: str | Sequence[str]) -> str:
        if not isinstance(s, str):
            s = "".join(str(c) for c in s)
        return "".join(c for c in s if c not in ", \t")

    for modality, strands in list(args.modality_strands.items()):
        strands = _normalize_strand_string(strands)
        args.modality_strands[modality] = strands
        n_bw = len(args.modality_to_bigwigs[modality])
        if len(strands) != n_bw:
            parser.error(
                f"strand string for modality '{modality}' has {len(strands)} "
                f"strand chars but there are {n_bw} bigwigs"
            )
        invalid = sorted({c for c in strands if c not in "+-."})
        if invalid:
            parser.error(
                f"strand string for '{modality}' contains invalid chars "
                f"{invalid}; allowed: '+', '-', '.'"
            )

    # Validate gene-LFC config consistency.
    if args.gene_loss_weight > 0:
        if not args.gtf:
            parser.error("--gene-loss-weight > 0 requires --gtf")
        if "rna_seq" not in args.modality_to_bigwigs:
            parser.error(
                "--gene-loss-weight > 0 requires rna_seq in --modality / config; "
                f"got modalities: {sorted(args.modality_to_bigwigs)}"
            )
        if "rna_seq" not in args.modality_strands:
            parser.error(
                "--gene-loss-weight > 0 requires per-track strand info for "
                "rna_seq. Pass --track-strands or set "
                "modalities.rna_seq.strand in the config."
            )

    # Validate gene-expression-eval config consistency.
    if getattr(args, "gene_expr_eval", False):
        if not (args.gene_expr_annotation or args.gtf):
            parser.error(
                "--gene-expr-eval requires an annotation with exon rows: pass "
                "--gene-expr-annotation (parquet/GTF) or --gtf."
            )
        if "rna_seq" not in args.modality_to_bigwigs:
            parser.error(
                "--gene-expr-eval requires rna_seq in --modality / config; "
                f"got modalities: {sorted(args.modality_to_bigwigs)}"
            )
        if "rna_seq" not in args.modality_strands:
            parser.error(
                "--gene-expr-eval requires per-track strand info for rna_seq "
                "(sense-strand matching). Pass --track-strands or set "
                "modalities.rna_seq.strand in the config."
            )

    if "--modality-weights" not in cli_flags:
        for mod, weight in _parse_weight_overrides(
            config_data.get("modality_weights", config_data.get("task_weights"))
        ).items():
            if mod not in args.modality_to_bigwigs:
                parser.error(f"Unknown modality in config task weights: {mod}")
            args.modality_weight_dict[mod] = weight

    if args.modality_weights:
        for item in args.modality_weights.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                parser.error("Each --modality-weights item must be 'modality:weight'")
            mod, weight = item.split(":", 1)
            mod = mod.strip()
            if mod not in args.modality_to_bigwigs:
                parser.error(f"Unknown modality in --modality-weights: {mod}")
            args.modality_weight_dict[mod] = float(weight.strip())

    args.is_multimodal = len(args.modalities) > 1

    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)
    tokens = list(argv) if argv is not None else sys.argv[1:]
    return postprocess_args(args, parser, tokens)

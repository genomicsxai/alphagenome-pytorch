---
name: alphagenome-finetuning
description: Fine-tune or transfer-learn AlphaGenome-PyTorch on custom genomic data — pick a mode (linear probe, LoRA, Locon, full), train on BigWig tracks with `agt finetune`, use adapters, delta checkpoints, multi-GPU/sequence parallelism, or the Python transfer API. Use when ADAPTING/TRAINING the model on new data, not when running predictions with the pretrained model.
---

# Fine-tuning AlphaGenome-PyTorch

Read **`docs/finetuning/`** for the full guide — it is the source of truth:

- `docs/finetuning/index.rst` — overview and quick start
- `docs/finetuning/cli.rst` — all CLI flags, YAML configs, delta checkpoints,
  multi-modality, multi-GPU
- `docs/finetuning/python_api.rst` — transfer API, heads, delta weights
- `docs/finetuning/adapters.rst` — linear probing, LoRA, Locon, IA3, merging
- `docs/finetuning/api_reference.rst` — API reference

`agt finetune --help` is the ground truth for flags.

## Quick orientation

Workflow: **load trunk → choose transfer mode → add heads for your tracks → train.**

Modes (`--mode`, default `lora`): `linear-probe` (heads only, fastest baseline),
`lora` (recommended), `locon` (adapts Conv1d layers), `lora+locon`, `full`,
`encoder-only`. Escalate only if the cheaper mode underfits.

```bash
agt finetune --mode lora \
    --genome hg38.fa \
    --modality atac --bigwig data/*.bw \
    --train-bed train.bed --val-bed val.bed \
    --pretrained-weights model.pth
```

`agt finetune` and `python scripts/finetune.py` are the same code path with the same
flags — use `agt` (it ships with the package; `scripts/` only exists in a clone).
For multi-GPU, `torchrun` needs a module target:
`torchrun --nproc_per_node=2 -m alphagenome_pytorch.cli finetune ...`

Modalities: `rna_seq`, `atac`, `dnase`, `procap`, `cage` (1bp + 128bp);
`chip_tf`, `chip_histone` (128bp only).

Optional data prep: `agt preprocess scale-bigwig --input *.bw --target 100M`
(depth-normalize) or `agt preprocess bigwig-to-mmap` (faster training I/O).

Gene-level RNA-seq (both off by default, see `docs/finetuning/cli.rst`):
- `--gene-loss-weight 0.1` adds the cross-track gene-LFC loss over gene bodies.
  Needs `--gtf`, `rna_seq` in `--modality`, and `--track-strands`.
- `--gene-expr-eval` reports exon-based gene-expression correlations each
  validation epoch (`rna_seq_gene_log_expr_pearson_*`). Needs an annotation
  **with exon rows** — `--gene-expr-annotation`, falling back to `--gtf`.
- Both `--gtf` and `--gene-expr-annotation` take parquet or GTF/GFF. Prefer
  parquet (`scripts/convert_gtf_to_parquet.py`): seconds vs minutes on startup,
  and one file with exon rows covers both features.

What a run writes: `<output-dir>/<run-name>` (default
`finetuning_output/<timestamp>`) gets `best_model.pth`, `checkpoint_epoch{N}.pth`
(every epoch), `config.json` and the CSV logs. **`--save-delta` is off by
default**, so a default run produces nothing shareable. `--no-full-checkpoint`
selects deltas-only; `--no-save-checkpoints` writes no weights at all.

Loading: full checkpoints and full exports are self-contained
(`agt predict --checkpoint X`); delta checkpoints, exported deltas and adapter
bundles also need `--model <base weights>`. `agt info <file>` or
`describe_checkpoint(path)` says which you have. See
[`docs/finetuning/checkpoints.rst`](../../../docs/finetuning/checkpoints.rst).

Gotchas:
- `--resolutions` defaults to `1` (1bp only); use `--resolutions 128` for
  `chip_tf`/`chip_histone`.
- `--locon-targets` is empty by default and **must** be set when Locon is enabled
  (e.g. `down_blocks.5`, or `down_blocks.4,down_blocks.5`).
- `--save-delta` works with every mode except `full`.
- Saving/exporting an adapter (`--save-delta`, `export_delta_weights`, or
  `agt adapters export`) works only from delta weights/checkpoints — merged
  adapters or a full-model fine-tune have no adapter weights to extract.
- There is no `--overlap-lowres` flag; it is computed as `overlap_highres // 128`.

For running predictions rather than training, see the `alphagenome-predictions`
skill and [`docs/alphagenome-usage.md`](../../../docs/alphagenome-usage.md).

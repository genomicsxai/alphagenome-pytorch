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

Gotchas:
- `--resolutions` defaults to `1` (1bp only); use `--resolutions 128` for
  `chip_tf`/`chip_histone`.
- `--locon-targets` is empty by default and **must** be set when Locon is enabled
  (e.g. `down_blocks.5`, or `down_blocks.4,down_blocks.5`).
- `--save-delta` works with every mode except `full`.
- There is no `--overlap-lowres` flag; it is computed as `overlap_highres // 128`.

For running predictions rather than training, see the `alphagenome-predictions`
skill and [`docs/alphagenome-usage.md`](../../../docs/alphagenome-usage.md).

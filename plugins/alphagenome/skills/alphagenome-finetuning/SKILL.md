---
name: alphagenome-finetuning
description: Fine-tune or transfer-learn AlphaGenome-PyTorch on custom genomic data — pick a mode (linear probe, LoRA, Locon, full), train on BigWig tracks with `agt finetune`, use adapters, delta checkpoints, multi-GPU/sequence parallelism, or the Python transfer API. Use when ADAPTING/TRAINING the model on new data, not when running predictions with the pretrained model.
---

# Fine-tuning AlphaGenome-PyTorch

Fine-tuning reuses the pretrained trunk as a sequence-representation extractor and
trains new heads (plus optionally adapters) on your tracks. The workflow is always:
**load trunk → choose transfer mode → add heads for your tracks → train.**

`agt finetune` is the supported entry point; it covers every mode below and takes CLI
flags or a YAML config (`--config`, CLI overrides YAML). Run `agt finetune --help` for
the authoritative flag list.

> **`agt finetune` ≡ `python scripts/finetune.py`.** They are the same code path with
> the same flags; `agt finetune --mode lora ...` and `python scripts/finetune.py --mode
> lora ...` are interchangeable. Prefer `agt` — it ships with the installed package,
> whereas `scripts/` only exists in a repo clone. The script remains as a thin shim, so
> older commands and configs keep working.

## 1. Pick a mode (`--mode`, default `lora`)

| Mode | Trains | Use when |
|------|--------|----------|
| `linear-probe` | heads only, trunk frozen | fastest; strong baseline; start here to sanity-check the data |
| `lora` | heads + LoRA on attention projections | **recommended default** for most tasks |
| `locon` | heads + Locon on Conv1d layers | the convolutional tower needs adapting (e.g. new assay statistics) |
| `lora+locon` | both adapter families | attention *and* conv tower need adapting |
| `full` | all parameters | large target dataset, willing to pay compute + risk of forgetting |
| `encoder-only` | encoder path only | specialised; see docs |

Escalate only if the cheaper mode underfits: `linear-probe` → `lora` → `lora+locon` / `full`.

## 2. Minimum viable command

```bash
agt finetune --mode lora \
    --genome hg38.fa \
    --modality atac --bigwig data/*.bw \
    --train-bed train.bed --val-bed val.bed \
    --pretrained-weights model.pth
```

Inputs: a genome FASTA, one or more BigWig signal tracks, train/val BED interval
files, and the pretrained checkpoint. No flag is strictly required at the argparse
level because any of them may come from `--config`; missing inputs fail at runtime
(e.g. `--bigwig is required (or provide modalities in --config)`).

Requires the finetuning extra: `pip install 'alphagenome-pytorch[finetuning]'`.

## 3. Modalities (`--modality`)

`rna_seq`, `atac`, `dnase`, `procap`, `cage` support **1bp and 128bp**.
`chip_tf`, `chip_histone` are **128bp only**.

Repeat `--modality`/`--bigwig` in pairs for multi-modality training.

## 4. Defaults worth knowing

`--resolutions` defaults to `1` (1bp only) — set `--resolutions 128` for
`chip_tf`/`chip_histone`, which have no 1bp output.

LoRA: `--lora-rank 8`, `--lora-alpha 16`, `--lora-targets q_proj,v_proj`.
Locon: `--locon-rank 4`, `--locon-alpha 1`, `--locon-targets` **empty by default —
you must set it** when Locon is enabled, e.g. `down_blocks.5` (last conv block) or
`down_blocks.4,down_blocks.5` (last two).

Training: `--epochs 10`, `--batch-size 1`, `--lr 1e-4`, `--weight-decay 0.1`,
`--warmup-steps 500`, `--lr-schedule cosine`, `--sequence-length 131072`,
`--output-dir finetuning_output`.

## 5. Common variations

**Delta checkpoints** (adapters + heads only, ~5–10MB vs ~1GB):
```bash
--save-delta [--no-full-checkpoint]
```
Works with every mode **except `full`** (which trains all parameters).

**Multi-GPU** — DDP, or sequence parallelism to split one long sequence across GPUs.
`torchrun` needs a module or script target, so use `-m`:
```bash
torchrun --nproc_per_node=2 -m alphagenome_pytorch.cli finetune \
    --sequence-parallel --overlap-highres 1024 ...
```
There is **no `--overlap-lowres` flag**; the low-res overlap is computed as
`overlap_highres // 128`.

**Memory pressure**: `--gradient-checkpointing`, `--gradient-accumulation-steps`,
`--batch-size 1`, `--dtype`/`--no-amp`.

**Tracking**: `--wandb` (`--wandb-project`, default `alphagenome-finetune`).

**Preparing BigWig data** (`agt preprocess`) — optional prep before training:
```bash
# Normalize signal depth across tracks to a common total (e.g. 100M)
agt preprocess scale-bigwig --input *.bw --output scaled/ --target 100M
# (add --dry-run to just print the scale factor)

# Convert BigWigs to a memory-mapped format for faster training I/O
agt preprocess bigwig-to-mmap --input *.bw --output mmap/ --chromosomes chr1 chr2
```
Training reads `.bw` directly, so this is only for depth normalization or I/O speed.

## 6. Python API (when the CLI doesn't fit)

```python
from alphagenome_pytorch import (
    AlphaGenome, TransferConfig, load_trunk, prepare_for_transfer,
)

model = AlphaGenome()
model = load_trunk(model, "model.pth")            # trunk only, excludes heads
model = prepare_for_transfer(model, TransferConfig(
    mode="lora",
    new_heads={"atac": {"modality": "atac", "num_tracks": 1}},
    lora_rank=8,
))
```

Re-exported from `alphagenome_pytorch.extensions.finetuning`: `TransferConfig`,
`MODALITY_CONFIGS`, `LoRA`, `Locon`, `IA3`, `train_epoch`, `validate`,
`save_checkpoint`, `export_delta_weights`, and datasets (`ATACDataset`,
`RNASeqDataset`, `MultimodalDataset`, `CachedGenome`).

## 7. Read the docs for depth

Don't guess flags or API shapes — these are authoritative and current:

- Overview & quick start: https://alphagenome-pytorch.readthedocs.io/en/latest/finetuning/index.html
- All CLI flags, YAML configs, delta checkpoints, multi-modality: https://alphagenome-pytorch.readthedocs.io/en/latest/finetuning/cli.html
- Transfer API, heads, saving/loading delta weights: https://alphagenome-pytorch.readthedocs.io/en/latest/finetuning/python_api.html
- Adapter families and merging: https://alphagenome-pytorch.readthedocs.io/en/latest/finetuning/adapters.html
- API reference: https://alphagenome-pytorch.readthedocs.io/en/latest/finetuning/api_reference.html

If working inside a clone of the repo, the same pages are `docs/finetuning/*.rst`.
`agt finetune --help` is the ground truth for flags.

To run *predictions* with a pretrained or finetuned model instead, see the
`alphagenome-predictions` skill. To score variants, see `agt score --help`.

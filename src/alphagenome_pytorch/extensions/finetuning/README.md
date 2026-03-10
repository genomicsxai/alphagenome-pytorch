# AlphaGenome Fine-tuning Extension

This extension provides tools for fine-tuning the AlphaGenome model on custom genomic datasets.

## Supported Modalities

| Modality | Description | Default Resolutions | Squashing |
|----------|-------------|---------------------|-----------|
| `atac` | ATAC-seq chromatin accessibility | 1bp, 128bp | No |
| `dnase` | DNase-seq chromatin accessibility | 1bp, 128bp | No |
| `procap` | PRO-cap transcription | 1bp, 128bp | No |
| `cage` | CAGE transcription | 1bp, 128bp | No |
| `rna_seq` | RNA-seq gene expression | 1bp, 128bp | Yes |
| `chip_tf` | ChIP-seq transcription factors | 128bp | No |
| `chip_histone` | ChIP-seq histone modifications | 128bp | No |

## Quick Start

Use the unified training entrypoint:

- `scripts/finetune.py` for single-modality and multi-modality runs
- CLI flags for quick experiments
- YAML config (`--config`) for reproducible experiments (requires `PyYAML`)

### Single-Modality Fine-tuning (Recommended)

```bash
python scripts/finetune.py --mode lora \
    --genome hg38.fa \
    --modality atac \
    --bigwig cell1.bw cell2.bw cell3.bw \
    --train-bed train_regions.bed \
    --val-bed val_regions.bed \
    --pretrained-weights alphagenome.pth \
    --resolutions 1 \
    --epochs 10 \
    --output-dir output/atac_finetune
```

### Multi-Modality Fine-tuning (CLI)

Repeat `--modality` and `--bigwig` pairs:

```bash
python scripts/finetune.py --mode lora \
    --genome hg38.fa \
    --pretrained-weights alphagenome.pth \
    --train-bed train.bed --val-bed val.bed \
    --modality atac --bigwig cell1_atac.bw cell2_atac.bw \
    --modality rna_seq --bigwig cell1_rna.bw \
    --modality-weights "atac:1.0,rna_seq:0.5"
```

### Multi-Modality Fine-tuning (YAML Config)

For complex setups, use `--config config.yaml` (CLI values override YAML values):

```yaml
# config.yaml
genome: /path/to/hg38.fa
pretrained_weights: /path/to/alphagenome.pth
train_bed: /path/to/train.bed
val_bed: /path/to/val.bed

# Output
output_dir: output/multitask
run_name: atac_rnaseq_experiment

# Training hyperparameters
epochs: 10
batch_size: 1
gradient_accumulation_steps: 4
lr: 1e-4
warmup_steps: 500
lr_schedule: cosine
num_segments: 8          # multinomial-loss segmentation (optional)
min_segment_size: 64     # optional floor for segment size

# LoRA configuration
lora_rank: 8
lora_alpha: 16
lora_targets: "q_proj,v_proj"

# Loss configuration
positional_weight: 5.0
count_weight: 1.0

# Modalities
modalities:
  atac:
    bigwig:
      - /path/to/cell1_atac.bw
      - /path/to/cell2_atac.bw
    resolutions: "1,128"  # Per-modality override (optional)
    task_weight: 1.0

  rna_seq:
    bigwig:
      - /path/to/cell1_rna.bw
    resolutions: "128"    # Per-modality override (optional)
    task_weight: 0.5

  chip_tf:
    bigwig:
      - /path/to/tf1.bw
      - /path/to/tf2.bw
    task_weight: 1.0
```

Run it with:

```bash
pip install pyyaml
python scripts/finetune.py --config config.yaml
```

Notes:
- Top-level `resolutions` is the global default.
- `modalities.<name>.resolutions` overrides the global value for that modality.
- `modalities.<name>.task_weight` sets per-modality loss weighting.
- CLI `--modality-weights` overrides YAML task weights when both are provided.
- `num_segments` and `min_segment_size` apply to both training and validation multinomial loss.

## Module Structure

```
finetuning/
├── heads.py        # create_finetuning_head() factory, ASSAY_TYPES config
├── training.py     # Training loop, loss computation, MODALITY_CONFIGS
├── datasets.py     # GenomicDataset, BigWig loading, compute_track_means
├── transfer.py     # LoRA, prepare_for_transfer, load_trunk
├── adapters.py     # LoRA adapter implementation
└── data_transforms.py  # Data augmentation utilities
```

## Key Components

### Creating a Fine-tuning Head

```python
from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head

# ATAC-seq head with 10 tracks at both resolutions
head = create_finetuning_head(
    assay_type='atac',
    n_tracks=10,
    resolutions=(1, 128),
    track_means=computed_means,  # Optional, for scaling
)

# ChIP-TF head (128bp only by default)
head = create_finetuning_head(
    assay_type='chip_tf',
    n_tracks=100,
)
```

### Training Configuration

```python
from alphagenome_pytorch.extensions.finetuning.training import (
    MODALITY_CONFIGS,
    create_lr_scheduler,
    compute_finetuning_loss,
)

# Get default config for a modality
config = MODALITY_CONFIGS['atac']
print(config.resolutions)  # (1, 128)
print(config.default_resolution_weights)  # {1: 1.0, 128: 1.0}

# Create LR scheduler with warmup
scheduler = create_lr_scheduler(
    optimizer,
    warmup_steps=500,
    total_steps=10000,
    schedule="cosine",
)
```

### Transfer Learning Setup

```python
from alphagenome_pytorch.extensions.finetuning.transfer import (
    prepare_for_transfer,
    TransferConfig,
    load_trunk,
    remove_all_heads,
)

# Load pretrained trunk
model = AlphaGenome()
model = load_trunk(model, "alphagenome.pth", exclude_heads=True)

# Freeze base model
for param in model.parameters():
    param.requires_grad = False

# Remove original heads
model = remove_all_heads(model)

# Apply LoRA adapters
config = TransferConfig(
    mode="lora",
    lora_targets=["q_proj", "v_proj"],
    lora_rank=8,
    lora_alpha=16,
)
model = prepare_for_transfer(model, config)
```

## Data Flow

```
Input BED regions + BigWig files
          │
          ▼
    GenomicDataset
    (loads sequences + signals)
          │
          ▼
    DataLoader with collate_fn
          │
          ▼
┌─────────┴──────────┐
│ AlphaGenome Trunk  │  ← Frozen or LoRA-adapted
│   model.encode()   │
└─────────┬──────────┘
          │
          ▼
    embeddings_dict
    {1: emb_1bp, 128: emb_128bp}
          │
          ▼
┌─────────┴──────────┐
│  GenomeTracksHead  │  ← Trainable
│  (per modality)    │
└─────────┬──────────┘
          │
          ▼
    predictions_dict
    {1: pred_1bp, 128: pred_128bp}
          │
          ▼
    multinomial_loss
    (positional + count components)
```

### Getting Embeddings

Use `model.encode()` to extract embeddings without running prediction heads:

```python
# Get embeddings (128bp only for efficiency)
emb = model.encode(dna_sequence, organism_index, resolutions=(128,))
# emb['embeddings_128bp']: (B, S//128, 3072) - NLC format

# For Conv1d heads, use NCL format (no transpose overhead)
emb = model.encode(dna_sequence, organism_index, resolutions=(128,), channels_last=False)
# emb['embeddings_128bp']: (B, 3072, S//128) - NCL format
```

Available embeddings:
- `embeddings_1bp`: (B, S, 1536) at 1bp resolution - only if `1` in resolutions
- `embeddings_128bp`: (B, S//128, 3072) at 128bp resolution - always computed
- `embeddings_pair`: (B, S//2048, S//2048, 128) for contact map prediction

## Memory Optimization

- **128bp only**: Use `resolutions=(128,)` to save ~100x memory vs 1bp
- **Gradient accumulation**: Set `--gradient-accumulation-steps 4` to simulate larger batches
- **LoRA**: Fine-tune with LoRA adapters (`--lora-rank 8`) instead of full model
- **Gradient checkpointing**: Enabled by default, disable with `--no-gradient-checkpointing`

## Testing

Run the unit tests:

```bash
# All finetuning tests
pytest tests/unit/test_finetuning_*.py -v

# Specific test files
pytest tests/unit/test_finetuning_heads.py -v
pytest tests/unit/test_finetuning_training.py -v
pytest tests/unit/test_finetune_multitask.py -v
```

## Loss Function

The fine-tuning uses a multinomial loss with two components:

1. **Positional loss**: Cross-entropy over positions (where is the signal?)
2. **Count loss**: Poisson regression on total counts (how much signal?)

Controlled by `--positional-weight` and `--count-weight` arguments.

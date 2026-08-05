# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaGenome-PyTorch is a PyTorch port of the AlphaGenome genomics model (originally in JAX). It predicts genomic tracks (chromatin accessibility, transcription, histone modifications) and 3D contact maps from DNA sequences.

- **Input**: One-hot encoded DNA (batch, 131072bp, 4 channels for ACGT)
- **Output**: Multi-resolution predictions (1bp and 128bp) for various genomic assays

## Common Commands

```bash
# Install
pip install -e alphagenome-pytorch

# Run unit tests (no JAX required)
pytest tests/unit/ -v

# Run PyTorch-only integration tests (no JAX required)
pytest tests/integration/ -v --torch-weights=model.pth

# Run JAX comparison integration tests (requires JAX checkpoint)
pytest tests/integration_jax/ -v \
    --jax-checkpoint=/path/to/checkpoint \
    --torch-weights=model.pth

# Run component-level JAX comparison tests
pytest tests/jax_comparison/ -v

# Run tests in parallel
pytest tests/ -n 4 --jax-checkpoint=...

# Convert JAX weights to PyTorch (includes track means)
agt convert --input jax_checkpoint --output model.pth
# equivalently (the script takes the checkpoint as a positional argument):
python scripts/convert_weights.py jax_checkpoint --output model.pth
```

## Architecture

The model follows an encoder-tower-decoder pattern with multi-resolution outputs:

```
DNA (B, 131072, 4)
    │
    ▼
SequenceEncoder: DnaEmbedder + 6 DownResBlocks + pooling
    │
    ▼
Trunk (B, 1024, 1536) @ 128bp  ─── intermediates dict for U-Net skips
    │
    ▼
TransformerTower: 9 blocks (PairUpdate on even blocks, AttentionBias, MHA, MLP)
    │                │
    ▼                ▼
Trunk          Pair Acts (B, 64, 64, 128)
    │                │
    ▼                ▼
SequenceDecoder    OutputPair → ContactMapsHead
    │
    ▼
Decoded (B, 131072, 768) @ 1bp
    │
    ▼
OutputEmbedders (128bp: 3072 dim, 1bp: 1536 dim)
    │
    ▼
GenomeTracksHead for each assay type
```

### Key Files

- `src/alphagenome_pytorch/model.py` - Main AlphaGenome class with SequenceEncoder, TransformerTower, SequenceDecoder
- `src/alphagenome_pytorch/attention.py` - RoPE, MHABlock, PairUpdateBlock, AttentionBiasBlock
- `src/alphagenome_pytorch/convolutions.py` - StandardizedConv1d, DownResBlock, UpResBlock
- `src/alphagenome_pytorch/heads.py` - GenomeTracksHead, ContactMapsHead, predictions_scaling
- `src/alphagenome_pytorch/embeddings.py` - OutputEmbedder, OutputPair
- `src/alphagenome_pytorch/sequence_parallel.py` - SequenceParallelism for multi-GPU inference and training
- `src/alphagenome_pytorch/extensions/finetuning/args.py` - CLI flag definitions for `agt finetune` / `scripts/finetune.py` (dependency-light: no torch import, so `--help` works without a full install)
- `src/alphagenome_pytorch/extensions/finetuning/runner.py` - finetuning orchestration: dataset/model construction, training loop dispatch, checkpointing. `scripts/finetune.py` is a thin shim over `runner.main()`

### Output Heads

| Head | Tracks | Resolutions | Notes |
|------|--------|-------------|-------|
| ATAC | 256 | 1bp, 128bp | Chromatin accessibility |
| DNase | 384 | 1bp, 128bp | |
| PRO-cap | 128 | 1bp, 128bp | |
| CAGE | 640 | 1bp, 128bp | |
| RNA-seq | 768 | 1bp, 128bp | `apply_squashing=True` |
| ChIP-TF | 1664 | 128bp only | |
| ChIP-Histone | 1152 | 128bp only | |
| Contact Maps | 28 | pair (S×S) | 3D chromatin |
| Splice Site | 5 classes | 1bp only | Donor+/Acceptor+/Donor-/Acceptor-/none |
| Splice Usage | 2×samples | 1bp only | proportion of RNA using each site |
| Splice Junctions | 2×tissues | pair (P×P) | junction read counts; RoPE over splice-site positions |

## Technical Notes

### JAX Compatibility
- Many implementation choices mirror the JAX reference for numerical validation
- Use `dtype_policy=DtypePolicy.mixed_precision()` for JAX-matching precision (bfloat16 compute)
- Default `DtypePolicy.default()` is `DtypePolicy.full_float32()` (works everywhere)
- Tests compare outputs with 1% relative tolerance to account for precision differences

### Custom Implementations
- `Pool1d`: SAME padding matching TensorFlow/JAX behavior
- `StandardizedConv1d`: Weight standardization with learned scaling
- Custom GELU: `sigmoid(1.702 * x) * x` to match JAX
- RoPE uses geometric frequency spacing

### Multi-organism Support
- Separate embeddings/heads per organism (index 0=human, 1=mouse)
- Track means provide per-organism scaling factors

### Sequence Parallelism
- `SequenceParallelism` splits the input sequence across GPUs instead of splitting the batch
- Works for both inference (`torch.no_grad()`) and training (`train_epoch_sequence_parallel`)
- Encoder and decoder run locally per rank; transformer runs globally after an all-gather of the trunk
- Embeddings returned are local to each rank (`S_local`, not full `S`)
- Enable with `agt finetune --sequence-parallel [--overlap-highres N]` (equivalently
  `python scripts/finetune.py`). There is no `--overlap-lowres` flag: the low-res
  overlap is computed as `overlap_highres // 128`. (The `SequenceParallelism` class
  itself does take an `overlap_lowres` argument.)

```bash
# torchrun needs a module/script target, so invoke the CLI via -m
torchrun --nproc_per_node=2 -m alphagenome_pytorch.cli finetune \
    --sequence-parallel --overlap-highres 1024 \
    --genome hg38.fa --modality atac --bigwig *.bw \
    --train-bed train.bed --val-bed val.bed --pretrained-weights model.pth
```

### Splice Fine-tuning
- Splice modalities (`splice_site`, `splice_usage`, `splice_junctions`) are handled
  separately from the generic bigwig-backed modalities: no `--bigwig` is required,
  they use `--star-junctions`/`--ssu` instead. A single `--modality` entry can
  comma-separate them (e.g. `--modality splice_site,splice_usage,splice_junctions`)
  to share one `--star-junctions`/`--ssu` group.
- `--gtf` is splice-site annotation (canonical splice sites, annotation-only, zero
  usage) — **distinct** from `--gene-gtf`, which feeds the unrelated gene-LFC-loss
  and gene-expression-eval features (`--gene-loss-weight`, `--gene-expr-eval`).
  Don't confuse the two; they're read by different code paths.
- Key finetuning flags: `--rope-init` (`truncated_normal` default matches the JAX
  pretrained distribution; `zeros` replicates the original buggy JAX init, for
  ablation only), `--junction-loss` (`original`/`normalized`/`sparse` cross-entropy
  variants), `--junction-position-source` (`annotated` uses STAR-derived positions;
  `predicted` derives them from the `splice_site` classification head's top-k sites,
  see `--junction-top-k`), `--pretrained-head-samples` (initialize head weights from
  specific pretrained output tracks), `--min-alpha-juncs` (junction depth threshold
  for the SSU loss).
- `scripts/compute_ssu.py` / `scripts/get_star_junctions.py`: standalone
  preprocessing scripts that derive splice site usage (SSU) and junction counts
  from STAR alignment output, upstream of `--ssu`/`--star-junctions`.
- LoCon (Conv1d LoRA, `--locon-rank`/`--locon-alpha`/`--locon-targets`) targets the
  CNN encoder's `down_blocks`, orthogonal to LoRA's transformer `q_proj`/`v_proj`
  targets — combine both via `--mode lora+locon`.

### Test Strategy
- Unit tests (`tests/unit/`): Fast, no JAX required, verify PyTorch components
- Integration tests (`tests/integration/`): PyTorch-only full model tests (backward pass, finetuning, variant scoring)
- JAX integration tests (`tests/integration_jax/`): Compare JAX vs PyTorch outputs, require JAX checkpoint
- JAX comparison tests (`tests/jax_comparison/`): Component-level JAX vs PyTorch parity
- Use `-k` to filter by organism or resolution: `pytest -k "human"` or `pytest -k "128-"`
- Sequence parallel tests use mocked dist on CPU; GPU/multi-GPU variants skip automatically when hardware is absent (`pytest -m slow` to include multi-GPU tests)

## Reference Documentation

- `ARCHITECTURE_COMPARISON.md` - Detailed JAX vs PyTorch comparison, component verification
- `tests/README.md` - Comprehensive test documentation

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
python scripts/convert_weights.py --input jax_checkpoint --output model.pth
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

### Test Strategy
- Unit tests (`tests/unit/`): Fast, no JAX required, verify PyTorch components
- Integration tests (`tests/integration/`): PyTorch-only full model tests (backward pass, finetuning, variant scoring)
- JAX integration tests (`tests/integration_jax/`): Compare JAX vs PyTorch outputs, require JAX checkpoint
- JAX comparison tests (`tests/jax_comparison/`): Component-level JAX vs PyTorch parity
- Use `-k` to filter by organism or resolution: `pytest -k "human"` or `pytest -k "128-"`

## Reference Documentation

- `ARCHITECTURE_COMPARISON.md` - Detailed JAX vs PyTorch comparison, component verification
- `tests/README.md` - Comprehensive test documentation

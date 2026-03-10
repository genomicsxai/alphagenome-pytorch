# AlphaGenome-Torch Test Suite

This directory contains pytest tests for validating the PyTorch implementation.

## Quick Start

```bash
# Install test dependencies
pip install -e ".[test]"

# Run unit tests (no JAX needed)
pytest tests/unit/ -v

# Run PyTorch-only integration tests (no JAX needed)
pytest tests/integration/ -v --torch-weights=model.pth

# Run JAX comparison integration tests (requires JAX checkpoint)
pytest tests/integration_jax/ -v \
    --jax-checkpoint=/path/to/checkpoint \
    --torch-weights=model.pth
```

## Test Organization

### Unit Tests (`tests/unit/`)

Unit tests verify PyTorch components work correctly without requiring JAX or full model.

| Test File | What It Tests |
|-----------|---------------|
| `test_heads.py` | `MultiOrganismLinear`, `GenomeTracksHead`, `ContactMapsHead`, `predictions_scaling` |
| `test_losses.py` | Loss functions: poisson, multinomial, MSE, cross-entropy |
| `test_metrics.py` | Pearson R, Spearman R, metrics computation |
| `test_attention.py` | RoPE, attention mechanisms |
| `test_adapters.py` | LoRA, Locon, IA3, Houlsby adapters |
| `test_checkpoint_roundtrip.py` | Model save/load roundtrip |
| `test_determinism.py` | Reproducibility with same seed |
| `test_finetuning_*.py` | Fine-tuning components |

### Integration Tests (`tests/integration/`)

PyTorch-only integration tests that verify end-to-end functionality.

| Test File | What It Tests | Requirements |
|-----------|---------------|--------------|
| `test_backward.py` | Backward pass, gradient flow | PyTorch model |
| `test_finetuning_pipeline.py` | Fine-tuning with mock data | `--torch-weights` |
| `test_variant_scoring.py` | Variant scoring pipeline | PyTorch model |
| `test_all_19_scorers.py` | All 19 variant scorers vs API | See below |

#### Requirements for `test_all_19_scorers.py`

This test compares PyTorch variant scoring against the AlphaGenome API. It requires:

1. **CUDA GPU** - Tests are skipped without CUDA
2. **PyTorch weights** - `--torch-weights=model.pth`
3. **Reference genome** - Set `ALPHAGENOME_FASTA_PATH` environment variable to your hg38.fa
4. **API results** - Either:
   - Pre-computed cache at `data/api_cache/variant_scores.pkl`, OR
   - Set `ALPHAGENOME_API_KEY` to fetch from API (results are cached)

Optional for full coverage:
- `ALPHAGENOME_GTF_PATH` - Gene annotations for gene-based scorers
- `ALPHAGENOME_POLYA_PATH` - PolyA annotations for PolyadenylationScorer
- `ALPHAGENOME_TRACK_METADATA_PATH` - Track metadata for name-based matching

```bash
# Example with minimal requirements:
ALPHAGENOME_FASTA_PATH=/path/to/hg38.fa \
pytest tests/integration/test_all_19_scorers.py -v --torch-weights=model.pth
```

### JAX Comparison Tests (`tests/integration_jax/`)

Integration tests that compare JAX vs PyTorch outputs.

| Test File | What It Tests | Requirements |
|-----------|---------------|--------------|
| `test_genomic_tracks.py` | All 7 genomic track heads | `--jax-checkpoint` |
| `test_contact_maps.py` | Contact maps head | `--jax-checkpoint` |
| `test_splicing.py` | Splice site heads | `--jax-checkpoint` |
| `test_gradient_ladder.py` | Gradient parity | `--jax-checkpoint` |
| `test_training_loss_parity.py` | Training loss parity | `--jax-checkpoint` |

### Component-Level JAX Comparison (`tests/jax_comparison/`)

Low-level JAX vs PyTorch component comparisons.

| Test File | What It Tests |
|-----------|---------------|
| `test_losses_jax.py` | Loss function parity |
| `test_gradients_jax.py` | Loss gradient parity |
| `test_layers_jax.py` | Layer intermediate outputs |
| `test_pure_functions_jax.py` | Pure functions (GELU, RoPE, pooling) |

## Output Types by Resolution

### Multi-Resolution Heads (1bp + 128bp)

| Head | Tracks | Special Notes |
|------|--------|---------------|
| **ATAC** | 256 | Chromatin accessibility |
| **DNase** | 384 | DNase-seq |
| **PRO-cap** | 128 | Nascent transcription |
| **CAGE** | 640 | Transcription start sites |
| **RNA-seq** | 768 | `apply_squashing=True` (power law expansion) |

### 128bp-Only Heads

| Head | Tracks | Special Notes |
|------|--------|---------------|
| **ChIP-TF** | 1664 | Transcription factor binding |
| **ChIP-Histone** | 1152 | Histone modifications |

### Pair-Based Head

| Head | Tracks | Shape | Special Notes |
|------|--------|-------|---------------|
| **Contact Maps** | 28 | (B, S, S, 28) | 3D chromatin contacts |

## Running Tests

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--jax-checkpoint` | None | Path to JAX checkpoint (required for `integration_jax/`) |
| `--torch-weights` | `model.pth` | Path to PyTorch weights (includes bundled track means) |
| `--rtol` | `0.05` | Relative tolerance (5%) |
| `--atol` | `1e-4` | Absolute tolerance |

### Examples

```bash
# Run all tests (requires JAX checkpoint)
pytest tests/ -v --jax-checkpoint=/path/to/checkpoint

# Run only unit tests (fast, no JAX)
pytest tests/unit/ -v

# Run PyTorch-only integration tests (no JAX needed)
pytest tests/integration/ -v --torch-weights=model.pth

# Run JAX comparison tests
pytest tests/integration_jax/ -v --jax-checkpoint=...
pytest tests/jax_comparison/ -v

# Run tests for a specific organism
pytest tests/integration_jax/ -v -k "human" --jax-checkpoint=...

# Adjust tolerance
pytest tests/integration_jax/ --rtol=0.02 --atol=1e-3 --jax-checkpoint=...

# Run in parallel (requires pytest-xdist)
pytest tests/ -n 4 --jax-checkpoint=...
```

### Using Markers

```bash
# Run only unit tests
pytest -m unit

# Run PyTorch-only integration tests
pytest -m integration --torch-weights=model.pth

# Run JAX comparison integration tests
pytest -m integration_jax --jax-checkpoint=...

# Run component-level JAX tests
pytest -m jax_comparison

# Run fine-tuning tests only
pytest -m finetuning --torch-weights=model.pth
```

## Tolerance Guidelines

The default tolerances (`rtol=5%`, `atol=1e-4`) account for:
- bfloat16 mixed-precision computation differences
- Different floating-point accumulation order between JAX and PyTorch
- Gradient accumulation through deep transformer layers (9 blocks)

## Adding New Tests

To add JAX comparison tests for a new head type:

1. Create `tests/integration_jax/test_newhead.py`
2. Follow the pattern in existing test files:
   ```python
   @pytest.mark.integration_jax
   @pytest.mark.jax
   class TestNewHead:
       HEAD_NAME = "new_head"
       NUM_TRACKS = 100

       @pytest.mark.parametrize("organism", ["human", "mouse"])
       def test_comparison(self, cached_predictions, tolerances, organism):
           # ... comparison logic
   ```
3. Add the output type mapping in `tests/integration_jax/conftest.py`

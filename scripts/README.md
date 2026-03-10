# Scripts

Utility scripts for AlphaGenome-PyTorch model conversion, training, and validation.

## Weight Conversion

### convert_weights.py
Convert JAX AlphaGenome checkpoint to PyTorch format. Bundles track_means automatically.

```bash
python scripts/convert_weights.py /path/to/jax/checkpoint --output model.pth
```

### extract_track_means.py
Extract track_means from JAX model metadata (requires JAX dependencies).

```bash
python scripts/extract_track_means.py --output track_means.pt
```

### extract_track_metadata.py
Extract full track metadata (names, ontology, tissue info) from JAX model.

```bash
python scripts/extract_track_metadata.py --output-file track_metadata.parquet
```

### validate_weight_mapping.py
Audit all parameter mappings between JAX and PyTorch models.

```bash
python scripts/validate_weight_mapping.py --jax-checkpoint /path/to/checkpoint
python scripts/validate_weight_mapping.py --jax-checkpoint /path/to/checkpoint --verbose
```

## Fine-tuning

### finetune.py
Unified training script supporting linear probing, LoRA, and full fine-tuning.

```bash
# Linear probing
python scripts/finetune.py --mode linear-probe \
    --genome hg38.fa \
    --modality atac --bigwig *.bw \
    --train-bed train.bed --val-bed val.bed \
    --pretrained-weights model.pth

# LoRA fine-tuning
python scripts/finetune.py --mode lora \
    --lora-rank 8 --lora-alpha 16 \
    --genome hg38.fa \
    --modality atac --bigwig *.bw \
    --train-bed train.bed --val-bed val.bed \
    --pretrained-weights model.pth

# Multi-GPU with DDP
torchrun --nproc_per_node=4 scripts/finetune.py --mode lora ...
```

## Data Preprocessing

### convert_bigwigs_to_mmap.py
Convert BigWig files to memory-mapped numpy arrays for fast random access.

```bash
# Single file
python scripts/convert_bigwigs_to_mmap.py \
    --bigwig signal.bw \
    --output-dir mmap_signals/signal

# Multiple files in parallel
python scripts/convert_bigwigs_to_mmap.py \
    --bigwig *.bw \
    --output-dir mmap_signals/ \
    --workers 8
```

### convert_borzoi_folds.py
Convert Borzoi sequence folds (~196kb) to AlphaGenome format (1Mb regions).

```bash
python scripts/convert_borzoi_folds.py \
    --input sequences_human.bed.gz \
    --output-dir data/alphagenome_folds
```

### convert_gtf_to_parquet.py
Convert GTF annotation files to Parquet for fast loading.

```bash
python scripts/convert_gtf_to_parquet.py \
    --input gencode.v49.annotation.gtf \
    --output gencode.v49.parquet
```

### preprocess_polya.py
Convert GENCODE polyA metadata to Parquet with gene_id linking.

```bash
python scripts/preprocess_polya.py \
    --metadata gencode.v46.metadata.PolyA_feature \
    --gtf gencode.v46.annotation.parquet \
    --output gencode.v46.polyAs.linked.parquet
```

## Validation & Benchmarking

### verify_model.py
Verify model loading and run a basic forward pass.

```bash
python scripts/verify_model.py
```

### compare_models.py
Compare JAX and PyTorch model outputs for numerical validation.

```bash
python scripts/compare_models.py /path/to/jax/checkpoint \
    --torch_weights model.pth
```

### simple_compare.py
Simplified JAX vs PyTorch comparison script.

```bash
python scripts/simple_compare.py /path/to/jax/checkpoint
```

### benchmark_performance.py
Benchmark model inference performance (timing, memory, GPU profiling).

```bash
python scripts/benchmark_performance.py --weights model.pth
```

## Demos

### demo_manual_extraction.py
Demonstrates manual sequence extraction and variant handling with JAX model.

```bash
python scripts/demo_manual_extraction.py /path/to/jax/checkpoint
```

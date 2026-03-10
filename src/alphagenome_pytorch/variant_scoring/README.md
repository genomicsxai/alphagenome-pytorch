# AlphaGenome PyTorch Variant Scoring

This module provides functionality for scoring the effects of genetic variants using the AlphaGenome PyTorch model. It implements the same variant scoring strategies as the official AlphaGenome API.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Variant Scorers](#variant-scorers)
6. [Resolution Selection](#resolution-selection)
7. [Working with Results](#working-with-results)
8. [Gene Annotations](#gene-annotations)
9. [Multi-Organism Support](#multi-organism-support)
10. [In-Silico Mutagenesis](#in-silico-mutagenesis)
11. [API Compatibility](#api-compatibility)
12. [Module Structure](#module-structure)

---

## Overview

The variant scoring module enables prediction of variant effects on:

- **Chromatin accessibility**: ATAC-seq, DNase-seq
- **Transcription**: CAGE, PRO-cap, RNA-seq
- **Histone modifications**: ChIP-seq histone marks
- **Transcription factor binding**: ChIP-seq TF
- **Splicing**: Splice site classification, usage, junctions
- **3D chromatin contacts**: Contact maps
- **Polyadenylation**: paQTLs (human only)

The scoring pipeline:
1. Extracts reference and alternate sequences for a variant
2. Runs model inference on both sequences
3. Applies scorer-specific aggregation methods
4. Returns per-track scores for downstream analysis

---

## Prerequisites

### Required Files

| File | Description |
|------|-------------|
| `model.pth` | PyTorch model weights |
| `track_means.pt` | Track mean values for normalization |
| `hg38.fa` / `mm10.fa` | Reference genome FASTA with `.fai` index |
| `gencode.v49.parquet` | Gene annotations (GTF also supported) |
| `gencode.v49.polyAs.parquet` | **[Optional]** PolyA sites for PolyadenylationScorer |
| `track_metadata.parquet` | Track metadata for tidy output |

### 1. Download Annotations
Get reference files from [GENCODE](https://www.gencodegenes.org/human/). We recommend v46 or newer.

```bash
mkdir -p data/annotations
cd data/annotations

# Gene Annotations
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.annotation.gtf.gz
gunzip gencode.v46.annotation.gtf.gz

# PolyA Annotations (for PolyadenylationScorer)
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.metadata.PolyA_feature.gz
gunzip gencode.v46.metadata.PolyA_feature.gz
cd ../..
```

### 2. Convert to Parquet (Recommended)
Parquet files load 50-100x faster than GTF/GFF.

```bash
# Convert Gene Annotations
python scripts/convert_gtf_to_parquet.py \
    --input data/annotations/gencode.v46.annotation.gtf \
    --output data/annotations/gencode.v46.annotation.parquet

# Process PolyA Annotations (creates linked parquet with Ensembl IDs)
# This is required for correct gene-level aggregation in PolyadenylationScorer
python scripts/preprocess_polya.py \
    --metadata data/annotations/gencode.v46.metadata.PolyA_feature \
    --gtf data/annotations/gencode.v46.annotation.parquet \
    --output data/annotations/gencode.v46.polyAs.linked.parquet
```

### 3. Track Metadata
Extract track metadata from the model weights file (if stored there) or download separately.

```bash
python scripts/extract_track_metadata.py --output track_metadata.parquet
```

---

## Quick Start

```python
import torch
from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.variant_scoring import (
    VariantScoringModel, Variant, Interval,
    CenterMaskScorer, OutputType, AggregationType,
    get_recommended_scorers,
)

# Load model (track_means are bundled in weights from convert_weights.py)
model = AlphaGenome()
model.load_state_dict(torch.load('model.pth'))

# Create scoring wrapper
scoring_model = VariantScoringModel(
    model,
    fasta_path='hg38.fa',
    gtf_path='gencode.v49.parquet',
    polya_path='gencode.v49.polyAs.parquet',  # Optional, for PolyadenylationScorer
    default_organism='human',
)
scoring_model.load_all_metadata('track_metadata.parquet')

# Define variant and interval
variant = Variant.from_str('chr22:36201698:A>C')
interval = Interval.centered_on('chr22', 36201698)

# Score with recommended scorers
scores = scoring_model.score_variant(
    interval, variant,
    scorers=get_recommended_scorers('human'),
    to_cpu=True,
)

# Convert to tidy DataFrame
df = scoring_model.tidy_scores(scores)
```

---

## Core Concepts

### Variants

Variants use VCF-style, 1-based positions:

```python
# From string (multiple formats supported)
variant = Variant.from_str('chr22:36201698:A>C')
variant = Variant.from_str('chr22_36201698_A_C', format='gtex')

# From components
variant = Variant('chr22', 36201698, 'A', 'C')

# Properties
variant.is_snv       # True for single nucleotide variants
variant.is_indel     # True for insertions/deletions
variant.start        # 0-based start position
variant.end          # 0-based exclusive end
```

### Intervals

Intervals use 0-based, half-open coordinates `[start, end)`:

```python
# Centered on position (131072bp default, required for AlphaGenome)
interval = Interval.centered_on('chr22', 36201698)
interval = Interval.centered_on('chr22', 36201698, width=131072)

# From coordinates
interval = Interval('chr22', 36136162, 36267234)

# From string
interval = Interval.from_str('chr22:36136162-36267234')

# Properties
interval.width       # Length in bp
interval.center      # Center position
interval.contains(variant)  # Check if variant is within interval
```

### Output Types

The model produces predictions for 11 assay types:

| OutputType | Description | Resolutions |
|------------|-------------|-------------|
| `ATAC` | ATAC-seq chromatin accessibility | 1bp, 128bp |
| `DNASE` | DNase-seq chromatin accessibility | 1bp, 128bp |
| `CAGE` | CAGE transcription start sites | 1bp, 128bp |
| `PROCAP` | PRO-cap nascent transcription | 1bp, 128bp |
| `RNA_SEQ` | RNA-seq gene expression | 1bp, 128bp |
| `CHIP_HISTONE` | ChIP-seq histone modifications | 128bp only |
| `CHIP_TF` | ChIP-seq transcription factors | 128bp only |
| `SPLICE_SITES` | Splice site classification | 1bp |
| `SPLICE_SITE_USAGE` | Splice site usage | 1bp |
| `SPLICE_JUNCTIONS` | Splice junction predictions | 1bp |
| `CONTACT_MAPS` | 3D chromatin contact maps | 128bp |

### Aggregation Types

Eight methods for comparing REF vs ALT predictions:

**Signed (directional) - can be positive or negative:**

| Type | Formula | Use Case |
|------|---------|----------|
| `DIFF_MEAN` | mean(ALT) - mean(REF) | Average change |
| `DIFF_SUM` | sum(ALT) - sum(REF) | Total change |
| `DIFF_SUM_LOG2` | sum(log2(ALT)) - sum(log2(REF)) | Log-scale position-wise |
| `DIFF_LOG2_SUM` | log2(sum(ALT)) - log2(sum(REF)) | **Recommended** - log fold change |

**Unsigned (non-directional) - always positive:**

| Type | Formula | Use Case |
|------|---------|----------|
| `L2_DIFF` | \|\|ALT - REF\|\|₂ | Euclidean distance |
| `L2_DIFF_LOG1P` | \|\|log1p(ALT) - log1p(REF)\|\|₂ | Log-scaled distance |
| `ACTIVE_MEAN` | max(mean(ALT), mean(REF)) | Maximum activity |
| `ACTIVE_SUM` | max(sum(ALT), sum(REF)) | Maximum total |

---

## Variant Scorers

### CenterMaskScorer

Spatial masking around variant position. The most versatile scorer for track-based outputs.

```python
scorer = CenterMaskScorer(
    requested_output=OutputType.ATAC,
    width=501,  # Supported: None, 501, 2001, 10001, 100001, 200001
    aggregation_type=AggregationType.DIFF_LOG2_SUM,
    resolution=None,  # None=auto, 1=1bp, 128=128bp
)
```

**Supported outputs**: ATAC, CAGE, DNASE, PROCAP, RNA_SEQ, CHIP_HISTONE, CHIP_TF, SPLICE_SITES, SPLICE_SITE_USAGE

### ContactMapScorer

Quantifies 3D chromatin contact disruption within a 1MB window centered on the variant.

```python
scorer = ContactMapScorer()  # Always non-directional
```

### GeneMaskLFCScorer

Gene-level log fold change using exon or gene body masks.

```python
from alphagenome_pytorch.variant_scoring.scorers.gene_mask import GeneMaskMode

scorer = GeneMaskLFCScorer(
    requested_output=OutputType.RNA_SEQ,
    mask_mode=GeneMaskMode.EXONS,  # EXONS or BODY
    resolution=128,  # Default 128bp for gene-level
)
# Returns one score per gene overlapping the interval
```

### GeneMaskActiveScorer

Gene-level active allele score (non-directional). Returns the maximum of the
**mean** predictions for REF and ALT within the gene mask, normalizing for
gene size.

```python
scorer = GeneMaskActiveScorer(
    requested_output=OutputType.RNA_SEQ,
    mask_mode=GeneMaskMode.EXONS,
    resolution=128,
)
# Returns: max(mean(ALT), mean(REF)) per gene
```

### GeneMaskSplicingScorer

Splicing impact using gene masks or center masks.

```python
scorer = GeneMaskSplicingScorer(
    requested_output=OutputType.SPLICE_SITES,
    width=None,  # None=gene mask, or 101, 1001, 10001
)
```

### SpliceJunctionScorer

Splice junction disruption scoring.

```python
scorer = SpliceJunctionScorer()
```

### PolyadenylationScorer

Polyadenylation QTL scoring using the **Borzoi paQTL method**. **Human only**.

Quantifies the maximum log2 fold change between proximal and distal polyadenylation
site (PAS) usage. See [Implementation Notes](#polyadenylationscorer-1) for the
full algorithm.

```python
scorer = PolyadenylationScorer(
    min_pas_count=2,      # Minimum PAS sites required per gene
    min_pas_coverage=0.8, # Fraction of gene's PAS in interval
)
```

For best results, initialize VariantScoringModel with `polya_path` pointing to a
GENCODE polyAs annotation file (e.g., `gencode.v49.polyAs.parquet`). This enables
gene-level proximal/distal PAS ratio scoring with 400bp windows around each PAS.
Without the annotation, the scorer falls back to peak detection.

### Recommended Scorers

Get pre-configured scorers matching the official API:

```python
# All recommended scorers for an organism
scorers = get_recommended_scorers('human')  # or 'mouse'

# View available presets
from alphagenome_pytorch.variant_scoring import RECOMMENDED_VARIANT_SCORERS
print(RECOMMENDED_VARIANT_SCORERS.keys())
```

---

## Resolution Selection

The AlphaGenome model produces predictions at two resolutions:
- **1bp resolution**: Fine-grained, position-level predictions
- **128bp resolution**: Binned predictions (each bin covers 128bp)

### When to Use Each Resolution

| Resolution | Best For | Trade-offs |
|------------|----------|------------|
| **1bp** | Narrow windows (≤2001bp), localized signals (ATAC/CAGE peaks) | Higher memory, more computation |
| **128bp** | Wide windows, gene-level scoring, efficiency | Less spatial precision |

### Auto-Selection (Default)

By default, CenterMaskScorer auto-selects resolution based on window width:
- `width ≤ 2001bp` → 1bp resolution
- `width > 2001bp` or `None` → 128bp resolution

### Explicit Control

You can override auto-selection:

```python
# Force 1bp resolution for detailed analysis
scorer = CenterMaskScorer(
    requested_output=OutputType.ATAC,
    width=501,
    aggregation_type=AggregationType.DIFF_LOG2_SUM,
    resolution=1,  # Explicit 1bp
)

# Force 128bp for efficiency
scorer = CenterMaskScorer(
    requested_output=OutputType.ATAC,
    width=501,
    aggregation_type=AggregationType.DIFF_LOG2_SUM,
    resolution=128,  # Explicit 128bp
)
```

### Resolution Constraints

Some output types have fixed resolutions:
- `CHIP_TF`, `CHIP_HISTONE`: 128bp only
- `CONTACT_MAPS`: 128bp only
- `SPLICE_SITES`, `SPLICE_SITE_USAGE`: 1bp only

---

## Working with Results

### VariantScore Objects

Each scorer returns `VariantScore` objects:

```python
score = scores[0]
score.variant        # Variant object
score.interval       # Interval object
score.scorer         # Scorer that produced this score
score.scores         # torch.Tensor of shape (num_tracks,)
score.gene_id        # Gene ID (for gene-centric scorers)
score.gene_name      # Gene name (for gene-centric scorers)
score.is_signed      # Whether scores are directional
```

### Converting to DataFrames

```python
# Using VariantScoringModel (includes loaded metadata)
df = scoring_model.tidy_scores(scores)

# Using standalone function
from alphagenome_pytorch.variant_scoring import tidy_scores
df = tidy_scores(scores, track_metadata=metadata_dict)
```

The tidy DataFrame has one row per track with columns:
- `variant_id`, `interval`
- `gene_id`, `gene_name`, `gene_type`, `gene_strand` (if applicable)
- `scorer`, `output_type`, `is_signed`
- `track_index`, `track_name`, `track_strand`
- `raw_score`, `quantile_score`
- Extended metadata: `ontology_curie`, `gtex_tissue`, `biosample_name`, etc.

### AnnData Output

For compatibility with the official API:

```python
from alphagenome_pytorch.variant_scoring import scores_to_anndata

adata = scores_to_anndata(scores, track_metadata=metadata_dict)
# adata.X: scores matrix
# adata.obs: variant/gene metadata
# adata.var: track metadata
```

### Track Metadata

```python
# Load all metadata at once
scoring_model.load_all_metadata('track_metadata.parquet')

# Load per output type
scoring_model.load_track_metadata(
    'atac_metadata.csv',
    OutputType.ATAC,
    organism='human',
)

# Access metadata
metadata = scoring_model.get_track_metadata(organism='human')
```

---

## Gene Annotations

```python
from alphagenome_pytorch.variant_scoring import GeneAnnotation

# Load from Parquet (recommended - 50-100x faster)
annotation = GeneAnnotation('gencode.v49.parquet')

# Load from GTF (requires pyranges)
annotation = GeneAnnotation('gencode.v49.gtf')

# Query genes in interval
genes = annotation.get_genes_in_interval(interval)
genes = annotation.get_genes_in_interval(
    interval,
    gene_types=['protein_coding', 'lncRNA'],
)

# Get gene info
info = annotation.get_gene_info('ENSG00000100342')

# Get masks for scoring
exon_mask = annotation.get_exon_mask(
    gene_id='ENSG00000100342',
    interval=interval,
    resolution=128,
    seq_length=1024,
)

gene_mask = annotation.get_gene_mask(
    gene_id='ENSG00000100342',
    interval=interval,
    resolution=128,
    seq_length=1024,
)
```

---

## Multi-Organism Support

The model supports Human (index 0) and Mouse (index 1):

```python
# Set default organism at initialization
scoring_model = VariantScoringModel(
    model,
    fasta_path='hg38.fa',
    gtf_path='gencode.v49.parquet',
    default_organism='human',  # or 'mouse', 0, 1
)

# Override per call
scores = scoring_model.score_variant(
    interval, variant, scorers,
    organism='mouse',  # or 1
)

# Some scorers are organism-specific
# PolyadenylationScorer: human only
```

---

## In-Silico Mutagenesis

Score all possible SNVs in a window around a position:

```python
# Score all SNVs in a 21bp window
ism_scores = scoring_model.score_ism_variants(
    interval=interval,
    center_position=36201698,  # 1-based
    scorers=[CenterMaskScorer(OutputType.ATAC, 501, AggregationType.DIFF_LOG2_SUM)],
    window_size=21,
    nucleotides='ACGT',  # Mutate to all 4 bases
    progress=True,
)

# Returns: list[list[VariantScore]] - [variant_idx][scorer_idx]
```

---

## API Compatibility

This implementation matches the official AlphaGenome gRPC API:

| Feature | PyTorch | Official API |
|---------|---------|--------------|
| 7 scorer types | ✅ | ✅ |
| 8 aggregation types | ✅ | ✅ |
| Width constraints | ✅ | ✅ |
| Multi-organism | ✅ | ✅ |
| tidy_scores() | ✅ | ✅ |
| Explicit resolution control | ✅ | N/A |
| ISM scoring | ✅ | ✅ |
| AnnData output | ✅ | ✅ |
| BODY vs EXONS mask mode | ✅ | ✅ (JAX) |

### Precision and Numerical Identity

The PyTorch model calculates outputs in **bfloat16** precision to match the JAX implementation (running on TPU). Due to the non-associativity of floating-point operations and differences between GPU and TPU kernels, **exact bitwise identity is not possible**.

However, we have verified that the PyTorch implementation is **numerically identical within machine precision limits**:
- **Max Absolute Difference**: < 2 ULPs (Units in Last Place) of bfloat16 resolution.
  - Example: For a score of 25,906.0, the bfloat16 resolution is 128.0. The observed difference is ~246.5 (1.92 steps).
- **Cosine Similarity**: > 0.999 for almost all tracks.
- **Log-Space Metrics**: `DIFF_LOG2_SUM` scorers may show absolute differences of 0.05-0.12 due to log-amplification of quantization noise.
- **ActiveSum Metrics**: Scorers aggregating unlogged values (e.g. `ACTIVE_SUM` for histone/CAGE) may show large absolute differences (e.g. 1500 vs 27000) due to accumulation precision differences between PyTorch (float32/64) and TPU (bfloat16). However, the **Cosine Similarity is > 0.999**, confirming architectural parity.

**Known Discrepancies**:
1. **PolyadenylationScorer**: Requires exact match of gene annotation versions. The API uses an internal version of GENCODE polyA sites combined with Ensembl gene IDs. Using public GENCODE files will result in significant scoring differences due to ID mismatch (XFAIL in tests).
2. **Float32 Inference**: Running the PyTorch model in `float32` will **increase** the absolute difference vs the JAX reference (which ran in bfloat16). Use `DtypePolicy.mixed_precision()` for maximum parity, or `DtypePolicy.full_float32()` (the default) for cleaner signals in production.


## Implementation Notes

This section documents the scoring algorithms in detail, including methodology
adopted from prior work (Borzoi, Orca) as described in the AlphaGenome paper.

### CenterMaskScorer

Standard center-mask scoring with 8 aggregation types. Creates a spatial mask
centered on the variant position, aggregates REF and ALT predictions within the
mask, then computes the difference (or max for active variants).

**Algorithm:**
1. Create boolean mask centered on variant (width: 501bp, 2001bp, etc.)
2. Apply mask to predictions at configured resolution (1bp or 128bp)
3. Compute aggregation (DIFF_LOG2_SUM recommended for most uses)

### GeneMaskLFCScorer

Gene-level log fold change using exon or gene body masks.

**Algorithm:**
```
score = log(mean(ALT within gene mask) + ε) - log(mean(REF within gene mask) + ε)
```

The log fold change is mathematically equivalent whether computed from mean or
sum (normalization factor cancels), but we follow the JAX implementation pattern.

### GeneMaskActiveScorer

Gene-level active allele score (non-directional). Reports the maximum mean
signal between REF and ALT alleles.

**Algorithm:**
```
score = max(mean(ALT within gene mask), mean(REF within gene mask))
```

Note: We compute MEAN (not sum) to normalize for gene size, following the JAX
implementation.

### GeneMaskSplicingScorer

Quantifies changes in splice site assignment probabilities or usage.

**Algorithm:**
```
score = max(|ALT - REF| within gene mask)
```

Uses max absolute difference (not mean) to capture the most impactful splice
site change within each gene.

### SpliceJunctionScorer

Scores splice junction disruption using junction-level predictions.

**Algorithm:**
1. Extract junction predictions (donor × acceptor × tissue)
2. Log-transform: `log(counts + 1e-7)`
3. Compute log fold change per junction
4. Report the junction with maximum total impact across tissues

### ContactMapScorer

Implements the **Orca method** (Zhou et al. 2022) for 3D chromatin contact
disruption scoring.

**Paper reference:** "For variants affecting 3D chromatin contacts, a method
similar to that used by Orca is employed for SNVs."

**Algorithm:**
1. Identify the 128bp bin containing the variant
2. Compute |ALT - REF| for entire contact map
3. Extract the row at the variant bin (contacts FROM this position)
4. Restrict to 1MB window centered on variant
5. Average over window positions for per-track scores

### PolyadenylationScorer

Implements the **Borzoi paQTL method** for polyadenylation scoring.

**Paper reference:** "Variant effects on polyadenylation are scored using a
method analogous to Borzoi's paQTL approach."

**Algorithm:**
1. Create 400bp windows around each polyadenylation site (strand-aware):
   - For + strand: 400bp upstream of PAS
   - For - strand: 400bp downstream of PAS
2. Aggregate RNA signal over each PAS window
3. Compute coverage ratio: `alt_coverage / ref_coverage` per PAS
4. For each possible proximal/distal split point k:
   - k_scaling = (total_PAS - k) / k
   - proximal = sum of coverage ratios for first k PAS
   - distal = sum of coverage ratios for remaining PAS
   - score = |log2(k_scaling × proximal / distal)|
5. Return maximum score across all split points

**Requirements:**
- Gene annotation file (GTF/Parquet)
- PolyA annotation file (GENCODE polyAs GTF/Parquet)
- At least 2 PAS sites per gene with 80% coverage in interval

### In-Silico Mutagenesis (ISM)

Systematic evaluation of all possible SNVs within a window.

**Algorithm:**
1. Generate all SNV variants (3 per position: original → A/C/G/T)
2. Score each variant with configured scorers
3. Construct contribution matrix (sequence_length × 4)
4. Mean-center: subtract mean of alternatives at each position

**Output interpretation:**
- Positive values indicate the reference base contributes to increased prediction
- Negative values indicate the reference base suppresses prediction

---

## Module Structure

```
variant_scoring/
├── __init__.py              # Public API exports
├── types.py                 # Variant, Interval, VariantScore, OutputType, AggregationType
├── sequence.py              # FastaExtractor, sequence encoding, variant application
├── annotations.py           # GeneAnnotation, exon/gene masks
├── aggregations.py          # compute_aggregation(), create_center_mask()
├── inference.py             # VariantScoringModel wrapper
├── visualization_utils.py   # Plotting utilities
└── scorers/
    ├── __init__.py
    ├── base.py              # BaseVariantScorer abstract class
    ├── center_mask.py       # CenterMaskScorer
    ├── contact_map.py       # ContactMapScorer
    ├── gene_mask.py         # GeneMaskLFCScorer, GeneMaskActiveScorer, GeneMaskMode
    ├── splicing.py          # GeneMaskSplicingScorer, SpliceJunctionScorer
    └── polyadenylation.py   # PolyadenylationScorer
```

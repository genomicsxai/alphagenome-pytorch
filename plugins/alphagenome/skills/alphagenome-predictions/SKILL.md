---
name: alphagenome-predictions
description: Run AlphaGenome-PyTorch to get genomic track predictions — via the `agt predict` CLI (single locus, BED regions, whole chromosomes, raw FASTA sequences, or per-gene count tables/AnnData), variant effect scoring (`agt score`), or the Python API. Covers picking a specific assay, cell type, or resolution, e.g. "get DNase predictions from GM12878 at 128bp", "write a wrapper for all K562 predictions", filtering tracks by metadata (biosample, assay, ontology, strand). Use when the task is about USING the model for inference/predictions, not developing the package.
---

# Getting predictions from AlphaGenome-PyTorch

The canonical usage guide is vendored with this plugin at:

```bash
GUIDE="${CLAUDE_PLUGIN_ROOT}/skills/alphagenome-predictions/reference/usage.md"
```

Read only the sections relevant to the task:

| Task | Read in `$GUIDE` |
|------|------------------|
| Write predictions to disk | `Command line: agt predict` |
| Score a variant's effect (SNV/VCF) | `Variant scoring: agt score` |
| Convert JAX weights / get a checkpoint | `Getting a checkpoint: agt convert` |
| Work with tensors in Python | `The 30-second version` and `Step 1` |
| Select an assay, cell type, or resolution | `Step 2`, `Step 3`, and `Recipes` |
| Look up exact track counts or metadata literals | `Step 2` and `Available metadata fields` |
| Diagnose padding, custom metadata, precision, or raw outputs | `Gotchas` |

Use a targeted search or section read instead of loading the whole guide. For example:

```bash
grep -n '^## ' "$GUIDE"
```

## Try the CLI first

If the task is "write predictions for these regions to disk", `agt predict` already
does it — no Python needed. One head per run; output format follows the input mode.

```bash
agt predict --model model.pth --output out/ --head rna_seq \
    --chromosomes chr20 --fasta hg38.fa --resolution 1 --crop-bp 16384 \
    --anndata gene_counts.h5ad --annotation gencode.v46.parquet \
    --aggregate-over exons --aggregate-func sum --gene-strand match
```

`--locus`/`--bed`/`--sequences`/`--chromosomes` are mutually exclusive. For a
finetuned model pass base weights as `--model` and the checkpoint as `--checkpoint`.
`agt predict --help` is the ground truth. `agt predict` is the same code path as the
older `scripts/predict_full_chromosome.py` shim — prefer `agt`, since `scripts/` only
exists in a repo clone.

Use the Python API below when you need tensors in-process or metadata-based track
selection beyond `--tracks`/`--track-names`.

## Quick orientation (Python API)

AlphaGenome emits thousands of tracks grouped into output heads (assays). Each
track is one channel in a tensor carrying metadata (cell type, assay, ontology,
strand). You rarely want a raw channel index — you want "DNase in GM12878". The
named-outputs API is the query layer over that metadata.

```python
import torch
from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor

model = AlphaGenome.from_pretrained("model.pth", device="cuda")
model.eval()

dna = sequence_to_onehot_tensor("ACGT" * 32768, device="cuda").unsqueeze(0)  # (1, 131072, 4)
out = model.predict(dna, organism_index=0, named_outputs=True)  # 0=human, 1=mouse

# "DNase predictions from GM12878 at 128bp"
dnase_gm = out.dnase.select(biosample_name="GM12878")[128].tensor  # (B, 1024, n_tracks)
```

- Input is one-hot DNA `(B, 131072, 4)`, ACGT order; the length is fixed at 131,072 bp.
- Filter on the head, then index by resolution: `head.select(...)[128].tensor`.
- Common filter fields: `biosample_name`, `assay_title`, `biosample_type`,
  `histone_mark`, `transcription_factor`, `ontology_curie`, `strand`.
- `.select()` matches strings **literally** and raises if nothing matches; pass
  `allow_empty=True` to get an empty tensor instead.
- Restrict work with `heads=` / `resolutions=` to skip expensive unused heads.

Explore the track catalog without weights — CLI first
(`agt info --heads`, `agt info --tracks dnase --filter biosample_name=K562` prints
matching track indices for `agt predict --tracks`), or in Python:

```python
from alphagenome_pytorch.named_outputs import TrackMetadataCatalog
cat = TrackMetadataCatalog.load_builtin("human")   # or "mouse"
tracks = cat.get_tracks("dnase", organism=0)
sorted({t.get("biosample_name") for t in tracks})
```

The vendored guide has the exact per-head track counts, the complete `assay_title`
and `biosample_type` value sets, and the padding rules — consult it rather than
guessing metadata strings.

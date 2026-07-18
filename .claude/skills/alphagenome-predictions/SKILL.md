---
name: alphagenome-predictions
description: Run AlphaGenome-PyTorch to get genomic track predictions — via the `agt predict` CLI (single locus, BED regions, whole chromosomes, raw FASTA sequences, or per-gene count tables/AnnData), variant effect scoring (`agt score`), or the Python API. Covers picking a specific assay, cell type, or resolution, e.g. "get DNase predictions from GM12878 at 128bp", "write a wrapper for all K562 predictions", filtering tracks by metadata (biosample, assay, ontology, strand). Use when the task is about USING the model for inference/predictions, not developing the package.
---

# Getting predictions from AlphaGenome-PyTorch

`docs/alphagenome-usage.md` is the canonical guide. Read only the relevant sections:

- Disk output: `Command line: agt predict`
- Variant effect scoring (SNV/VCF): `Variant scoring: agt score`
- Convert JAX weights / obtain a checkpoint: `Getting a checkpoint: agt convert`
- Python tensors and input shapes: `The 30-second version` and `Step 1`
- Assay/cell-type/resolution selection: `Step 2`, `Step 3`, and `Recipes`
- Exact counts and metadata literals: `Step 2` and `Available metadata fields`
- Padding, custom metadata, precision, or raw outputs: `Gotchas`

**Try the CLI first** — `agt predict` writes predictions to disk without any Python:

```bash
agt predict --model model.pth --output out/ --head dnase \
    --locus chr1:1000000-1131072 --fasta hg38.fa --resolution 128
```

Input modes (mutually exclusive): `--locus` (one interval), `--bed` (many regions),
`--chromosomes` (whole chromosomes, tiled), `--sequences` (raw FASTA → NPZ). Add
`--anndata FILE --annotation GTF` for a per-gene count table (AnnData); add `--gene-strand match` for RNA-seq so antisense tracks don't inflate counts. `agt predict`
is the same code path as the `scripts/predict_*.py` shims — prefer `agt`, which ships
with the package. See `agt predict --help`.

For **variant effect scoring**, use `agt score` (not `predict`):

```bash
agt score --model model.pth --fasta hg38.fa --variant "chr22:36201698:A>C" --output scores.tsv
```

`--vcf` for batches; `--scorer recommended` (default) or a comma-separated subset;
gene-centric scorers need `--gtf`. See the guide's `Variant scoring: agt score`.

Use the Python API when you need tensors in-process or metadata-based selection:

- Load: `AlphaGenome.from_pretrained("model.pth", device=...)`.
- Predict with metadata: `model.predict(dna, organism_index, named_outputs=True)`
  where `dna` is one-hot `(B, 131072, 4)` and `organism_index` is 0=human / 1=mouse.
- Select tracks by biology, then index by resolution:
  `out.dnase.select(biosample_name="GM12878")[128].tensor`.
- Filter fields include `biosample_name`, `assay_title`, `biosample_type`,
  `histone_mark`, `transcription_factor`, `ontology_curie`, `strand`.

Explore available tracks without weights: `agt info --heads`,
`agt info --tracks dnase --filter biosample_name=K562` (prints track indices for
`agt predict --tracks`), or in Python `TrackMetadataCatalog.load_builtin("human")`.

For the deeper API reference see `docs/named_outputs.rst`; for package
development conventions see `CLAUDE.md`.

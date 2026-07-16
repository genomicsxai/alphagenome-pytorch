# Using AlphaGenome-PyTorch to get predictions

This is the canonical guide for *running* AlphaGenome and pulling out predictions
for a specific assay, cell type, or resolution. The installable plugin carries a
CI-checked vendored mirror so it remains usable outside a repository clone. This
guide is agent-agnostic — any human or coding agent can follow it. (For *developing*
the package, see [`CLAUDE.md`](https://github.com/genomicsxai/alphagenome-pytorch/blob/main/CLAUDE.md).
For the full named-outputs API reference, see
[`docs/named_outputs.rst`](https://github.com/genomicsxai/alphagenome-pytorch/blob/main/docs/named_outputs.rst).)

The mental model: AlphaGenome emits **thousands of tracks** grouped into a
handful of **output heads** (assays). Each track is one channel in a tensor
and carries metadata (cell type, assay, ontology, strand, …). You almost never
want a raw channel index — you want "DNase in GM12878" or "everything in K562".
The named-outputs API turns that metadata into a query layer over the tensors.

**Reach for the CLI first.** If you just need predictions written to disk for some
regions, `agt predict` already does it — no Python required. Drop to the Python API
when you need tensors in-process, or track selection richer than `--tracks` allows.

## Contents

- [Command line: `agt predict`](#command-line-agt-predict)
- [The 30-second version (Python)](#the-30-second-version-python)
- [Step 1 — Inputs and shapes](#step-1--inputs-and-shapes)
- [Step 2 — Output heads and resolutions](#step-2--output-heads-assays-and-resolutions)
- [Step 3 — Selecting tracks by metadata](#step-3--selecting-tracks-by-metadata)
- [Recipes](#recipes-the-users-actual-asks)
- [Gotchas](#gotchas)

## Command line: `agt predict`

`agt` is the supported entry point (installed with the package). One head per run;
output format follows the input mode.

```bash
# One locus → BigWig
agt predict --model model.pth --output out/ --head dnase \
    --locus chr1:1000000-1131072 --fasta hg38.fa --resolution 128

# Many regions from a BED → BigWig (merged)
agt predict --model model.pth --output out/ --head atac \
    --bed regions.bed --fasta hg38.fa

# Whole chromosomes, tiled → BigWig
agt predict --model model.pth --output out/ --head rna_seq \
    --chromosomes chr20,chr21 --fasta hg38.fa --crop-bp 16384

# Raw FASTA sequences → NPZ per sequence
agt predict --model model.pth --output out/ --head atac --sequences seqs.fa

# Per-gene count table → AnnData (.h5ad)
agt predict --model model.pth --output out/ --head rna_seq \
    --chromosomes chr20 --fasta hg38.fa --resolution 1 --crop-bp 16384 \
    --anndata gene_counts.h5ad --annotation gencode.v46.parquet \
    --aggregate-over exons --aggregate-func sum
```

Notes:

- Input modes are mutually exclusive: `--locus`, `--bed`, `--sequences`, `--chromosomes`.
- Short `--locus`/`--bed` regions are padded with real reference flanks; long ones are
  center-cut unless you pass `--tile`. FASTA `--sequences` must match the window size
  exactly, or need `--tile`.
- `--anndata` aggregates signal per gene (over `exons` by default, or `gene-body`) into
  a genes × tracks matrix. It requires `--annotation` (GTF/parquet), which needs exon
  rows when aggregating over exons.
- Finetuned models: pass the base weights as `--model` and the finetuned checkpoint as
  `--checkpoint` (plus `--transfer-config` if it isn't embedded).
- Useful extras: `--tracks` / `--track-names`, `--resolution {1,128}`, `--organism {0,1}`,
  `--batch-size`, `--device`, `--dtype-policy`, `--compile`. See `agt predict --help`.

Install extras: `pip install 'alphagenome-pytorch[inference]'`, or
`'alphagenome-pytorch[inference-anndata]'` for `--anndata`.

**Relationship to the scripts.** `agt predict` is the same code path as the older
`scripts/predict_locus.py` and `scripts/predict_full_chromosome.py`; those remain as
thin shims for backwards compatibility. Prefer `agt` — the scripts are not part of the
installed package and only work from a repo clone.

## The 30-second version (Python)

```python
import torch
from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor

model = AlphaGenome.from_pretrained("model.pth", device="cuda")
model.eval()

# DNA → one-hot (L, 4); model wants a batch (B, 131072, 4)
seq = "ACGT" * 32768                       # exactly 131_072 bp
dna = sequence_to_onehot_tensor(seq, device="cuda").unsqueeze(0)

# named_outputs=True attaches the built-in track metadata catalog
out = model.predict(dna, organism_index=0, named_outputs=True)  # 0=human, 1=mouse

# "DNase predictions from GM12878 at 128bp resolution"
dnase_gm = out.dnase.select(biosample_name="GM12878")[128].tensor
# shape: (B, 1024, n_matching_tracks)
```

That's the whole pattern. The rest of this guide explains the pieces so you can
build wrappers for arbitrary queries.

## Step 1 — Inputs and shapes

- **Input**: one-hot DNA of shape `(B, 131072, 4)`, channels in `ACGT` order.
  Use `sequence_to_onehot_tensor(seq)` (returns `(L, 4)`) and `.unsqueeze(0)`
  for the batch axis. `N`/unknown bases encode as all-zeros.
- **Sequence length is fixed at 131,072 bp.** Shorter inputs must be padded;
  longer loci must be tiled (see
  [`scripts/predict_locus.py`](https://github.com/genomicsxai/alphagenome-pytorch/blob/main/scripts/predict_locus.py)).
- **`organism_index`**: `0` = human, `1` = mouse. Pass an `int` (broadcast over
  the batch) or a `(B,)` long tensor. Metadata and head widths differ per
  organism, so this also selects which catalog is used for named outputs.

To fetch real genomic sequence from a FASTA, use
`GenomeSequenceProvider` from
`alphagenome_pytorch.extensions.inference.full_chromosome`:

```python
from alphagenome_pytorch.extensions.inference.full_chromosome import GenomeSequenceProvider
genome = GenomeSequenceProvider("hg38.fa", chromosomes={"chr1"})
seq = genome.fetch("chr1", start, start + 131072)   # returns one-hot np.ndarray
dna = torch.from_numpy(seq).float().unsqueeze(0)
```

## Step 2 — Output heads (assays) and resolutions

`model.predict(...)` returns a dict (or `NamedOutputs` when
`named_outputs=True`) keyed by head name. Each head maps a **resolution** to a
tensor:

| Head           | Resolutions | Raw dim | Real (human) | Notes                    |
|----------------|-------------|---------|--------------|--------------------------|
| `atac`         | 1bp, 128bp  | 256     | 167          | chromatin accessibility  |
| `dnase`        | 1bp, 128bp  | 384     | 305          |                          |
| `procap`       | 1bp, 128bp  | 128     | 12           |                          |
| `cage`         | 1bp, 128bp  | 640     | 546          |                          |
| `rna_seq`      | 1bp, 128bp  | 768     | 667          | `apply_squashing=True`   |
| `chip_tf`      | 128bp       | 1664    | 1617         | TF binding               |
| `chip_histone` | 128bp       | 1152    | 1116         | histone marks            |
| `contact_maps` | pair (S×S)  | 28      | 28           | 3D contacts              |

**Raw dim** is the fixed tensor width; **Real (human)** is the non-padding track
count in the human catalog (`named_outputs=True` returns only these). The gap is
padding — see Gotchas. Mouse has far fewer real tracks (e.g. `procap` has **0**
real mouse tracks, so mouse PRO-cap is entirely padding); query
`TrackMetadataCatalog.load_builtin("mouse")` for exact mouse counts.

The model also emits three **splice** heads — `splice_sites` (raw 5),
`splice_site_usage` (734 human / 180 mouse), and `splice_junctions` (367) — valid
in the `heads=` filter but with a specialized per-tissue / junction-mask output
structure that doesn't follow the simple `head.select(...)[res]` pattern below.
See [`docs/named_outputs.rst`](https://github.com/genomicsxai/alphagenome-pytorch/blob/main/docs/named_outputs.rst)
and [`src/alphagenome_pytorch/heads.py`](https://github.com/genomicsxai/alphagenome-pytorch/blob/main/src/alphagenome_pytorch/heads.py)
if you need them.

Tensor layout is **channels-last by default**: `(B, positions, tracks)`. At
128bp a 131,072 bp input gives 1,024 positions; at 1bp it gives 131,072.

To compute only the heads/resolutions you need (much faster), pass them through:

```python
out = model.predict(dna, 0, named_outputs=True,
                     heads=("dnase",), resolutions=(128,))
```

## Step 3 — Selecting tracks by metadata

With `named_outputs=True` each head is a `NamedOutputHead`. Filtering happens on
the head (resolution-independent, since metadata is shared), and you index by
resolution at the end to get the tensor:

```python
head = out.dnase                      # NamedOutputHead
sel  = head.select(biosample_name="GM12878")   # filtered NamedOutputHead
ntt  = sel[128]                       # NamedTrackTensor at 128bp
ntt.tensor                            # (B, 1024, n_tracks) torch.Tensor
ntt.tracks                            # tuple[TrackMetadata], one per channel
ntt.to_dataframe()                    # pandas view of the selected metadata
```

`.select()` raises if nothing matches (pass `allow_empty=True` to get an empty
tensor instead). Matching modes:

```python
head.select(biosample_name="K562")                 # exact match
head.select(biosample_name=["K562", "GM12878"])     # any-of (list/set/tuple)
head.select(strand=None)                            # field missing / None
head.select(predicate=lambda t: "liver" in t.track_name.lower())  # arbitrary
```

If you only want channel indices or a boolean mask (e.g. for masking a loss
instead of slicing), use `.indices(...)` / `.mask(...)` with the same kwargs.

### Available metadata fields

Every track exposes these (access as `track.<field>` or `track.get("field")`):

`track_index`, `track_name`, `output_name`, `organism`, `strand`,
`ontology_curie`, `biosample_name`, `biosample_type`, `biosample_life_stage`,
`assay_title`, `data_source`, `gtex_tissue`, `histone_mark`,
`transcription_factor`, `endedness`, `genetically_modified`, `nonzero_mean`.

Useful value sets (human catalog, real tracks only — exact strings, `.select`
matches literally):
- `biosample_type` (complete set): `cell_line`, `tissue`, `primary_cell`,
  `in_vitro_differentiated_cells`, `organoid`
- `assay_title` (complete set): `DNase-seq`, `ATAC-seq`, `TF ChIP-seq`,
  `Histone ChIP-seq`, `polyA plus RNA-seq`, `total RNA-seq`, `hCAGE`, `LQhCAGE`,
  `PRO-cap`, `in situ Hi-C`, `Dilution Hi-C`, `Micro-C`
- `biosample_name`: 714 distinct cell types/tissues (human; 179 for mouse),
  including `K562` (333 tracks) and `GM12878` (125 tracks)

To explore what's available without running the model, load the catalog directly:

```python
from alphagenome_pytorch.named_outputs import TrackMetadataCatalog
cat = TrackMetadataCatalog.load_builtin("human")     # or "mouse", or None for both
cat.outputs(0)                                        # head names with metadata
tracks = cat.get_tracks("dnase", organism=0)
sorted({t.get("biosample_name") for t in tracks})     # all DNase cell types
```

## Recipes (the user's actual asks)

**"Get DNase predictions from GM12878 at 128bp resolution"**

```python
out = model.predict(dna, 0, named_outputs=True, heads=("dnase",), resolutions=(128,))
dnase_gm = out.dnase.select(biosample_name="GM12878")[128].tensor
```

**"Write a wrapper to get all the K562 cell predictions from AlphaGenome"**

```python
def k562_predictions(model, dna, organism_index=0, resolution=128):
    """Return {head_name: NamedTrackTensor} for every K562 track."""
    out = model.predict(dna, organism_index, named_outputs=True)
    result = {}
    for head_name in out.heads():
        head = out[head_name]
        if resolution not in head:                 # e.g. chip_tf has no 1bp
            continue
        sel = head.select(biosample_name="K562", allow_empty=True)[resolution]
        if sel.num_tracks:
            result[head_name] = sel
    return result
```

`out.select(biosample_name="K562")` does the same cross-head sweep in one call,
returning a `{(head, resolution): NamedTrackTensor}` dict.

**"All histone ChIP tracks for H3K27ac in any tissue"**

```python
out.chip_histone.select(histone_mark="H3K27ac", biosample_type="tissue")[128]
```

**Map a track back to its biology**

```python
ntt = out.dnase.select(biosample_name="GM12878")[128]
for t in ntt.tracks:
    print(t.track_index, t.track_name, t.assay_title, t.ontology_curie)
```

## Gotchas

- **Padding tracks**: raw head dims are padded to a fixed width (DNase is 384
  channels but only 305 are real human tracks; PRO-cap 128 → 12). Padding tracks
  have `track_name == "Padding"`. `named_outputs=True` strips them by default,
  matching the JAX reference. Pass `include_padding=True` to keep them.
- **Custom metadata**: instead of the built-in catalog, load your own with
  `model.load_track_metadata("my_tracks.parquet")` (parquet/csv/tsv), or build
  one via `TrackMetadataCatalog.from_dataframe(df)` and
  `model.set_track_metadata_catalog(catalog)`.
- **Precision**: default is float32. For JAX-matching bfloat16 compute, load with
  `dtype_policy=DtypePolicy.mixed_precision()` (from `alphagenome_pytorch.config`).
- **Raw dict access**: `named_outputs=False` (default) returns the plain
  `{head: {resolution: tensor}}` dict — use it when you don't need metadata.
  From a `NamedOutputs`, `.as_dict()` gives you the underlying raw dict back.

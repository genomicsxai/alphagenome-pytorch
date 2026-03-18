# Metadata API Comparison: JAX vs PyTorch

This document compares the metadata management APIs between the JAX (alphagenome_research) and PyTorch (alphagenome-pytorch) implementations.

## 1. Loading Metadata

### JAX
```python
from alphagenome_research.model import dna_model

model = dna_model.create_from_kaggle('all_folds')
metadata = model.output_metadata(dna_model.Organism.HOMO_SAPIENS)

# Access by output type enum
for output_type in dna_model.OutputType:
    num_tracks = len(metadata.get(output_type))
```

### PyTorch
```python
from alphagenome_pytorch.named_outputs import TrackMetadataCatalog

catalog = TrackMetadataCatalog.load_builtin('human')

# List available outputs
for output_name in catalog.outputs(organism=0):
    tracks = catalog.get_tracks(output_name, organism=0)
    print(f"{output_name}: {len(tracks)} tracks")
```

---

## 2. Making Predictions with Metadata

### JAX
```python
# Predictions already include metadata via the model wrapper
predictions = model.predict_interval(
    interval,
    requested_outputs={dna_model.OutputType.RNA_SEQ, dna_model.OutputType.DNASE},
    ontology_terms=['EFO:0001187'],  # Pre-filter by ontology
)

# Access metadata on the prediction object
predictions.rna_seq.metadata  # pandas DataFrame
```

### PyTorch
```python
from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.named_outputs import NamedOutputs, TrackMetadataCatalog

model = AlphaGenome.from_pretrained("path/to/weights.pth")
catalog = TrackMetadataCatalog.load_builtin('human')

# Get raw outputs
outputs = model(dna_tensor, organism_index=0)

# Wrap with metadata
named = NamedOutputs.from_raw(outputs, organism='human', catalog=catalog)

# Access metadata on track tensor
named.rna_seq[1].tracks  # tuple of TrackMetadata
named.rna_seq[1].to_dataframe()  # pandas DataFrame
```

---

## 3. Filtering by Metadata Columns

### JAX
```python
# Filter by assay type (uses pandas-style boolean indexing)
output_rna_seq = predictions.rna_seq.filter_tracks(
    (predictions.rna_seq.metadata['Assay title'] == 'total RNA-seq').values
)

# Filter by histone mark
output_chip_histone = predictions.chip_histone.filter_tracks(
    (predictions.chip_histone.metadata['histone_mark'] == 'H3K27ac').values
)

# Multiple conditions with & operator
output_chip_tf = predictions.chip_tf.filter_tracks(
    (
        (predictions.chip_tf.metadata['transcription_factor'] == 'CTCF')
        & (predictions.chip_tf.metadata['genetically_modified'].isnull())
    ).values
)
```

### PyTorch
```python
# Filter by assay type (uses kwargs)
output_rna_seq = named.rna_seq[1].select(assay_title='total RNA-seq')

# Filter by histone mark
output_chip_histone = named.chip_histone[128].select(histone_mark='H3K27ac')

# Multiple conditions (combined in single select call)
# Use field=None to match missing/null fields
output_chip_tf = named.chip_tf[128].select(
    transcription_factor='CTCF',
    genetically_modified=None  # Match tracks where field is missing/None
)

# Alternative: use list for "in" matching
output_chip_tf = named.chip_tf[128].select(
    transcription_factor=['CTCF', 'FOXA1']  # Match any
)
```

---

## 4. Strand Filtering

### JAX
```python
# Built-in strand filter methods
predictions.rna_seq.filter_to_unstranded()
predictions.rna_seq.filter_to_positive_strand()
predictions.rna_seq.filter_to_nonpositive_strand()  # negative or unstranded
predictions.splice_junctions.filter_to_strand('+')
```

### PyTorch
```python
# Use select() with strand field
named.rna_seq[1].select(strand='.')      # Unstranded
named.rna_seq[1].select(strand='+')      # Positive
named.rna_seq[1].select(strand=['-','.']) # Non-positive (negative or unstranded)
```

---

## 5. Tissue Filtering

### JAX
```python
# Built-in tissue filter method
predictions.splice_junctions.filter_by_tissue('Artery_Aorta')
```

### PyTorch
```python
# Use select() with gtex_tissue field
named.splice_sites_junction[1].select(gtex_tissue='Artery_Aorta')
```

---

## 6. Getting Indices / Boolean Masks

### JAX
```python
# Uses numpy boolean arrays directly
mask = (predictions.chip_tf.metadata['transcription_factor'] == 'CTCF').values
filtered = predictions.chip_tf.filter_tracks(mask)

# Index-based filtering with np.eye
chip_alt_ref.filter_tracks(np.eye(1, 6, 0, dtype=bool)[0])  # First track only
```

### PyTorch
```python
# Get indices
indices = named.chip_tf[128].indices(transcription_factor='CTCF')

# Get boolean mask
mask = named.chip_tf[128].mask(transcription_factor='CTCF')

# Use mask in loss computation
loss = ((preds - targets) ** 2 * mask).mean()

# Manual index selection
tensor = named.chip_tf[128].tensor[..., indices]
```

---

## 7. Arithmetic on Filtered Tracks

### JAX
```python
# Direct arithmetic on prediction objects
diff = prediction.alternate.rna_seq - prediction.reference.rna_seq

# Combined filter + arithmetic
chip_diff = (
    prediction.alternate.chip_histone.filter_to_nonpositive_strand()
    - prediction.reference.chip_histone.filter_to_nonpositive_strand()
)
```

### PyTorch
```python
# Must access tensor explicitly
alt_rna = named_alt.rna_seq[1].tensor
ref_rna = named_ref.rna_seq[1].tensor
diff = alt_rna - ref_rna

# Combined filter + arithmetic (more verbose)
alt_chip = named_alt.chip_histone[128].select(strand=['-', '.']).tensor
ref_chip = named_ref.chip_histone[128].select(strand=['-', '.']).tensor
chip_diff = alt_chip - ref_chip
```

**Note**: PyTorch `NamedTrackTensor` also supports direct arithmetic (`+`, `-`, `*`, `/`, `abs`)
which preserves metadata from the left operand.

---

## 8. Finetuning Metadata Setup

### JAX
```python
import pandas as pd
from alphagenome_research.model import dna_model
from alphagenome_research.model.metadata import metadata as metadata_lib

TRACK_METADATA = pd.DataFrame(
    data=[
        ['RNA_SEQ', 'UBERON:0000948 total RNA-seq', '+', '/tmp/file1.bigWig'],
        ['RNA_SEQ', 'UBERON:0000948 total RNA-seq', '-', '/tmp/file2.bigWig'],
        ['DNASE', 'EFO:0005337 DNase-seq', '.', '/tmp/file3.bigWig'],
    ],
    columns=['output_type', 'name', 'strand', 'file_path'],
)

def build_output_metadata(track_metadata):
    metadata = {}
    for output_type, df_group in track_metadata.groupby('output_type'):
        output_type = dna_model.OutputType[str(output_type)]
        metadata[output_type.name.lower()] = df_group
    return metadata_lib.AlphaGenomeOutputMetadata(**metadata)

output_metadata = {
    dna_model.Organism.HOMO_SAPIENS: build_output_metadata(TRACK_METADATA)
}
```

### PyTorch
```python
from alphagenome_pytorch.named_outputs import TrackMetadataCatalog, TrackMetadata

# Option 1: Load from file
catalog = TrackMetadataCatalog.from_file(
    "track_metadata.parquet",
    default_organism=0,
)

# Option 2: Build programmatically
catalog = TrackMetadataCatalog()
catalog.add_tracks(
    "rna_seq",
    [
        TrackMetadata(0, "rna_seq", 0, "UBERON:0000948 total RNA-seq",
                      extras={"strand": "+", "file_path": "/tmp/file1.bigWig"}),
        TrackMetadata(1, "rna_seq", 0, "UBERON:0000948 total RNA-seq",
                      extras={"strand": "-", "file_path": "/tmp/file2.bigWig"}),
    ],
    organism=0,
)
catalog.add_tracks(
    "dnase",
    [
        TrackMetadata(0, "dnase", 0, "EFO:0005337 DNase-seq",
                      extras={"strand": ".", "file_path": "/tmp/file3.bigWig"}),
    ],
    organism=0,
)
```

---

## Design Rationale

### Why metadata and tensor are bundled (`NamedTrackTensor`)

`NamedTrackTensor` keeps the tensor and its track metadata in lockstep.
After any `.select()` call, the returned object is guaranteed to have
matching tensor channels and metadata — no manual index tracking needed.

```python
# After filtering, tensor and metadata are always in sync
filtered = named.rna_seq[128].select(strand='+')
filtered.tensor  # already sliced to matching channels
filtered.tracks  # already filtered — same length as tensor's track axis
```

The alternative — separating metadata and tensors — would require users
to manually track indices and apply them to both, which is error-prone:

```python
# ❌ Hypothetical separated design — easy to get out of sync
indices = metadata.select(strand='+')
tensor = raw_tensors['rna_seq'][128][..., indices]   # manual slicing
# Did we apply the same indices? Are they still aligned?
```

### Why metadata is resolution-independent (`NamedOutputHead`)

Track metadata (names, ontology, biosample, strand, etc.) does not change
between resolutions — only the tensor's sequence dimension differs.
`NamedOutputHead` exposes shared metadata so you don't need to pick a
resolution for metadata-only operations:

```python
head = named.rna_seq
head.tracks           # resolution-independent
head.num_tracks       # same at 1bp and 128bp
head.to_dataframe()   # no resolution needed
```

### `.select()` works at both levels

Both orderings produce identical results:

```python
# Filter first, then pick resolution
named.rna_seq.select(strand='+')[128].tensor

# Pick resolution first, then filter
named.rna_seq[128].select(strand='+').tensor
```

`NamedOutputHead.select()` returns a new `NamedOutputHead` with filtered
tensors at all resolutions. `NamedTrackTensor.select()` returns a new
`NamedTrackTensor` with the filtered tensor at one resolution.

---

## Summary: Feature Comparison


| Feature | JAX | PyTorch | Notes |
|---------|-----|---------|-------|
| Load built-in metadata | `model.output_metadata()` | `TrackMetadataCatalog.load_builtin()` | Similar |
| Filter by field | `.filter_tracks(mask)` | `.select(**criteria)` | PyTorch more ergonomic |
| Filter null/missing | `metadata['col'].isnull()` | `.select(field=None)` | PyTorch cleaner |
| Access metadata field | `metadata['field']` | `track.field` | PyTorch uses attribute access |
| Safe field access | N/A | `track.get('field', default)` | PyTorch only |
| Check field exists | N/A | `track.has('field')` | PyTorch only |
| Multiple conditions | pandas `&` operator | kwargs | PyTorch cleaner |
| Strand filtering | `.filter_to_strand('+')` | `.select(strand='+')` | PyTorch uses generic `.select()` |
| Tissue filtering | `.filter_by_tissue()` | `.select(gtex_tissue=...)` | PyTorch uses generic `.select()` |
| Get indices | N/A (use numpy) | `.indices()` | PyTorch has dedicated method |
| Get mask | N/A (use numpy) | `.mask()` | PyTorch has dedicated method |
| To DataFrame | `.metadata` property | `.to_dataframe()` | Both work |
| Arithmetic | Direct on objects | Direct on objects | Identical |
| List organisms | N/A | `.organisms` property | PyTorch only |
| List outputs | N/A | `.outputs(organism)` | PyTorch only |
| Check catalog entry | N/A | `.has_tracks()` | PyTorch only |
| Cross-head filtering | N/A | `.select(**criteria)` | PyTorch only |
| Allow empty results | N/A | `.select(allow_empty=True)` | PyTorch only |
| Load from DataFrame | N/A | `.from_dataframe(df)` | PyTorch only |

---

## PyTorch-Only Features

### Direct attribute access on TrackMetadata
```python
# Access any metadata field directly (no need to use .extras dict)
track.ontology_curie          # Direct attribute access
track.biosample_type          # Works for any extras field
track.get('field', 'default') # Safe access with fallback
track.has('field')            # Check if field exists and is not None
```

### Null/missing field filtering
```python
# Match tracks where a field is missing or None
tracks.select(genetically_modified=None)

# Combine with other filters
tracks.select(transcription_factor='CTCF', genetically_modified=None)
```

### Strand and tissue filtering via `.select()`
```python
tracks.select(strand='+')           # Positive strand
tracks.select(strand='-')           # Negative strand
tracks.select(strand='.')           # Unstranded
tracks.select(strand=['+', '-'])    # Stranded (positive or negative)
tracks.select(strand=['-', '.'])    # Non-positive strand
tracks.select(gtex_tissue='Artery_Aorta')  # Tissue filtering
```

### Arithmetic operators
```python
diff = alt_tracks - ref_tracks
sum_tracks = tracks1 + tracks2
scaled = tracks * 2.0
normalized = tracks / tracks.tensor.max()
negated = -tracks
absolute = abs(tracks)
```

### Allow empty results
```python
filtered = tracks.select(tissue='nonexistent', allow_empty=True)
print(filtered.num_tracks)  # 0
print(filtered.shape)       # torch.Size([1, 100, 0])
```

### Cross-head `select()` method
```python
tissue_tracks = named.select(biosample_type='tissue')
for (output_name, resolution), tracks in tissue_tracks.items():
    print(f"{output_name}@{resolution}bp: {tracks.num_tracks} tracks")
```

### `from_dataframe` class method
```python
import pandas as pd
df = pd.DataFrame({
    'track_name': ['track_0', 'track_1'],
    'output_type': ['atac', 'atac'],
    'organism': [0, 0],
    'biosample_type': ['tissue', 'cell_line'],
})
catalog = TrackMetadataCatalog.from_dataframe(df)
```

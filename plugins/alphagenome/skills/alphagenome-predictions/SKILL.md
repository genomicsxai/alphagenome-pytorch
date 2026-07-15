---
name: alphagenome-predictions
description: Run AlphaGenome-PyTorch to get genomic track predictions for a specific assay, cell type, or resolution — e.g. "get DNase predictions from GM12878 at 128bp", "write a wrapper for all K562 predictions", filtering tracks by metadata (biosample, assay, ontology, strand). Use when the task is about USING the model for inference/predictions, not developing the package.
---

# Getting predictions from AlphaGenome-PyTorch

The full guide is bundled with this plugin. **Read it before writing code:**

```bash
cat "${CLAUDE_PLUGIN_ROOT}/skills/alphagenome-predictions/reference/usage.md"
```

It covers input shapes, the output heads and their exact track counts, selecting
tracks by metadata, worked recipes, and gotchas (padding, precision).

## Quick orientation

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

Explore the track catalog without loading weights:

```python
from alphagenome_pytorch.named_outputs import TrackMetadataCatalog
cat = TrackMetadataCatalog.load_builtin("human")   # or "mouse"
tracks = cat.get_tracks("dnase", organism=0)
sorted({t.get("biosample_name") for t in tracks})
```

The bundled guide has the exact per-head track counts, the complete `assay_title`
and `biosample_type` value sets, and the padding rules — consult it rather than
guessing metadata strings.

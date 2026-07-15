---
name: alphagenome-predictions
description: Run AlphaGenome-PyTorch to get genomic track predictions for a specific assay, cell type, or resolution — e.g. "get DNase predictions from GM12878 at 128bp", "write a wrapper for all K562 predictions", filtering tracks by metadata (biosample, assay, ontology, strand). Use when the task is about USING the model for inference/predictions, not developing the package.
---

# Getting predictions from AlphaGenome-PyTorch

Read **`docs/alphagenome-usage.md`** for the full guide. It is the
source-of-truth for running the model and selecting tracks by metadata.

Quick orientation:

- Load: `AlphaGenome.from_pretrained("model.pth", device=...)`.
- Predict with metadata: `model.predict(dna, organism_index, named_outputs=True)`
  where `dna` is one-hot `(B, 131072, 4)` and `organism_index` is 0=human / 1=mouse.
- Select tracks by biology, then index by resolution:
  `out.dnase.select(biosample_name="GM12878")[128].tensor`.
- Filter fields include `biosample_name`, `assay_title`, `biosample_type`,
  `histone_mark`, `transcription_factor`, `ontology_curie`, `strand`.

Explore available tracks without weights via
`TrackMetadataCatalog.load_builtin("human")`.

For the deeper API reference see `docs/named_outputs.rst`; for package
development conventions see `CLAUDE.md`.

# AlphaGenome PyTorch

A PyTorch port of [AlphaGenome](https://www.nature.com/articles/s41586-025-10014-0), the DNA sequence model from Google DeepMind that predicts hundreds of genomic tracks at single base-pair resolution from sequences up to 1M bp.

We strive to make it an accessible, readable, and hackable implementation — for integrating into existing PyTorch pipelines, fine-tuning on custom datasets, and building on top of.

## Installation

Installation from PyPI:

```bash
pip install alphagenome-pytorch
```

Installation from repo: 

```bash
pip install git+https://github.com/genomicsxai/alphagenome-pytorch
```

For fine-tuning (incl. BigWig data loading):

```bash
pip install alphagenome-pytorch[finetuning]  # adds pyBigWig, pyfaidx
```

## Quick Start

```python
from alphagenome_pytorch import AlphaGenome

# Load pretrained model
model = AlphaGenome.from_pretrained('alphagenome.pt', device='cuda')

# Create one-hot encoded DNA sequence in NLC format (batch=1, length=131072, channels=4)
# Channels: A=0, C=1, G=2, T=3
sequence = np.random.randint(0, 4, size=(1, 131072))
dna_onehot = torch.tensor(np.eye(4)[sequence], dtype=torch.float32)

# Inference (handles dtype casting, returns float32 outputs)
outputs = model.predict(dna_onehot, organism_index=0)  # organism: 0=human, 1=mouse

# outputs['atac'][1]   -> (B, 131072, 256) ATAC at 1bp
# outputs['atac'][128] -> (B, 1024, 256) ATAC at 128bp
# outputs['contact_maps'] -> (B, 28, 64, 64) 3D contact
```

The weights for this port are [available on Hugging Face](https://huggingface.co/gtca/alphagenome_pytorch).

## Extracting Embeddings

Use `model.encode()` to get embeddings without running prediction heads — useful for
building custom heads or analyzing representations:

```python
# Get embeddings (128bp only for efficiency)
emb = model.encode(dna_onehot, organism_index=0, resolutions=(128,))
emb['embeddings_128bp']  # (B, 1024, 3072) at 128bp
```

## Fine-tuning

Train a new head on your data with frozen trunk (linear probing) or with LoRA adapters:

```python
from alphagenome_pytorch import AlphaGenome, TransferConfig, load_trunk, prepare_for_transfer

# Load trunk, freeze, add custom heads
model = AlphaGenome()
model = load_trunk(model, 'alphagenome.pt')
model = prepare_for_transfer(model, TransferConfig(
    mode='lora',
    new_heads={'atac': {'modality': 'atac', 'num_tracks': 1}},
    lora_rank=8,
))
```

The easiest way to start with fine-tuning is to use [`scripts/finetune.py`](scripts/finetune.py) that implements a flexible CLI interface:

```bash
# LoRA fine-tuning
python scripts/finetune.py --mode lora --lora-rank 8 \
    --genome hg38.fa --modality atac --bigwig *.bw \
    --train-bed train.bed --val-bed val.bed \
    --pretrained-weights alphagenome.pt

# Multi-GPU
torchrun --nproc_per_node=4 scripts/finetune.py --mode lora ...
```

See [`examples/notebooks/finetune_linear_probe.ipynb`](examples/notebooks/finetune_linear_probe.ipynb) for an example of linear probing on ATAC-seq data.

## Numerical Parity with JAX

This port is validated against [the original JAX model](https://github.com/google-deepmind/alphagenome_research), including per-head and full forward pass output comparisons as well as loss values and gradients.

See a compiled [ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md) for some technical details.

## Model Outputs

| Head | Tracks | Resolutions | Description |
|------|--------|-------------|-------------|
| atac | 256 | 1bp, 128bp | Chromatin accessibility |
| dnase | 384 | 1bp, 128bp | DNase-seq |
| procap | 128 | 1bp, 128bp | Transcription initiation |
| cage | 640 | 1bp, 128bp | 5' cap RNA |
| rnaseq | 768 | 1bp, 128bp | RNA expression |
| chip_tf | 1664 | 128bp | TF binding |
| chip_histone | 1152 | 128bp | Histone modifications |
| contact_maps | 28 | 64×64 | 3D chromatin contacts |
| splice_sites | 4 | 1bp | Splice site classification (D+, A+, D−, A−) |
| splice_junctions | 734 | pairwise | Junction read counts (367 tissues × 2 strands) |
| splice_site_usage | 734 | 1bp | Fraction of transcripts using splice site |

See more information about model outputs [in the official AlphaGenome documentation](https://www.alphagenomedocs.com/exploring_model_metadata.html).

## Example Notebooks

- [Demo](examples/notebooks/alphagenome_pytorch_demo.ipynb) — Basic inference and JAX comparison
- [Variant Scoring](examples/notebooks/variant_scoring.ipynb) — Effect prediction
- [In Silico Mutagenesis](examples/notebooks/in_silico_mutagenesis.ipynb) — ISM analysis
- [TAL1 Mutation Example](examples/notebooks/TAL1_variant_effect_and_ISM.ipynb) - TAL1 variant effect and ISM (Figure 6 from AlphaGenome)
- [Fine-tuning](examples/notebooks/finetune_linear_probe.ipynb) — ATAC-seq linear probing
- [Fine-tuning](examples/notebooks/finetune_encoder_only.ipynb) — MPRA (encoder-only)

## Citation

```bibtex
@article{avsec2026alphagenome,
  title={Advancing regulatory variant effect prediction with AlphaGenome},
  author={Avsec, {\v{Z}}iga and Latysheva, Natasha and Cheng, Jun and Novati, Guido and Taylor, Kyle R and Ward, Tom and Bycroft, Clare and Nicolaisen, Lauren and Arvaniti, Eirini and Pan, Joshua and others},
  journal={Nature},
  volume={649},
  number={8099},
  pages={1206--1218},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```

<details>
<summary>bioRxiv preprint</summary>

```bibtex
@article{avsec2025alphagenome,
    title = {AlphaGenome: advancing regulatory variant effect prediction with a unified DNA sequence model},
    author = {Avsec, {\v Z}iga and Latysheva, Natasha and Cheng, Jun and ...},
    year = {2025},
    journal = {bioRxiv},
    doi = {10.1101/2025.06.25.661532}
}
```

</details>

## Acknowledgements

We acknowledge [Phil Wang](https://gitlab.com/lucidrains), [Miquel Anglada-Girotto](https://github.com/MiqG), and [Xinming Tu](https://github.com/XinmingTu) as developers of an older AlphaGenome PyTorch port unrelated to this repo. Note that the PyPI namespace is now linked to this repo.

## License

This project is a port of the [google-deepmind/alphagenome_research](https://github.com/google-deepmind/alphagenome_research) repository licensed under the Apache License, Version 2.0:

>Copyright 2026 Google LLC

The model parameters, output, and any derivatives thereof remain subject to [Google DeepMind’s AlphaGenome Model Terms](https://deepmind.google.com/science/alphagenome/model-terms).

This port is licensed under the Apache License, Version 2.0 (Apache 2.0):

>Copyright 2026 Danila Bredikhin, Martin Kjellberg, Christopher Zou, Alejandro Buendia, Xinming Tu, Anshul Kundaje

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this except in compliance with the License.
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


Finetuning
==========

We can unlock the utility of AlphaGenome for new datasets with fine-tuning / transfer learning.
We use the pretrained trunk to extract rich sequence representations, then add custom heads for specific prediction tasks.

Overview
--------

The typical finetuning workflow is:

1. **Load pretrained weights** (trunk only, excluding heads)
2. **Configure transfer mode** (full, linear probing, LoRA, Locon, IA3 — or combine adapter modes)
3. **Add custom heads** for your target tracks
4. **Train** using the target tracks

Quick Start
-----------

.. code-block:: bash

   # Linear probing (frozen backbone, fastest)
   python scripts/finetune.py --mode linear-probe \
       --genome hg38.fa \
       --modality atac --bigwig *.bw \
       --train-bed train.bed --val-bed val.bed \
       --pretrained-weights model.pth

   # LoRA finetuning (recommended)
   python scripts/finetune.py --mode lora \
       --lora-rank 8 --lora-alpha 16 \
       --genome hg38.fa \
       --modality atac --bigwig *.bw \
       --train-bed train.bed --val-bed val.bed \
       --pretrained-weights model.pth

   # Full finetuning (all parameters)
   python scripts/finetune.py --mode full \
       --genome hg38.fa \
       --modality atac --bigwig *.bw \
       --train-bed train.bed --val-bed val.bed \
       --pretrained-weights model.pth


.. toctree::
   :maxdepth: 2
   :caption: Finetuning Topics:

   cli
   python_api
   adapters
   api_reference

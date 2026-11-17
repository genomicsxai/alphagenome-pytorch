Finetuning
==========

We can unlock the utility of AlphaGenome for new datasets with fine-tuning / transfer learning.
With the extensive functionality for model fine-tuning, we can use the pretrained trunk to extract rich sequence representations, extending it with low-rank adapters and custom heads for specific prediction tasks.

Overview
--------

The typical finetuning workflow is:

1. **Load pretrained weights** (trunk only, excluding heads)
2. **Configure transfer mode** (full, linear probing, LoRA, Locon, IA3 — or combine adapter modes)
3. **Add custom heads** for your target tracks
4. **Train** using the target tracks
5. **Load or share the result** — see :doc:`checkpoints`

.. tip::

   Adding ``--save-delta`` to a training run enables writing
   a ~10MB artifact you can publish. Without it, a run writes only ~1GB full
   checkpoints, which cannot be packaged as an adapter bundle.

Quick Start
-----------

.. code-block:: bash

   # Linear probing (frozen backbone, fastest)
   agt finetune --mode linear-probe \
       --genome hg38.fa \
       --modality atac --bigwig *.bw \
       --train-bed train.bed --val-bed val.bed \
       --pretrained-weights model.pth

   # LoRA finetuning (recommended)
   agt finetune --mode lora \
       --lora-rank 8 --lora-alpha 16 \
       --genome hg38.fa \
       --modality atac --bigwig *.bw \
       --train-bed train.bed --val-bed val.bed \
       --pretrained-weights model.pth \
       --save-delta

   # Full finetuning (all parameters)
   agt finetune --mode full \
       --genome hg38.fa \
       --modality atac --bigwig *.bw \
       --train-bed train.bed --val-bed val.bed \
       --pretrained-weights model.pth


.. toctree::
   :maxdepth: 2
   :caption: Finetuning Topics:

   cli
   checkpoints
   python_api
   adapters
   api_reference

.. seealso::

   :doc:`../serving/adapters` — packaging a fine-tune as an adapter bundle,
   publishing it to Hugging Face, and serving one or many over a shared trunk.

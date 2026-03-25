Performance
===========

This page covers techniques for making AlphaGenome inference and training faster
and more memory-efficient.

.. contents:: On this page
   :local:
   :depth: 2

``torch.compile``
-----------------

``torch.compile`` is the highest-impact optimization for both inference and training.

The first forward pass triggers compilation and will be slower. 
All subsequent calls use the compiled graph.

.. code-block:: python

   import torch
   from alphagenome_pytorch import AlphaGenome

   model = AlphaGenome.from_pretrained('model.pth', device='cuda')
   model.eval()
   model = torch.compile(model)

   # First call is slow (compilation). Subsequent calls are fast.
   outputs = model.predict(dna_onehot, organism_idx=0)

Some scripts like ``finetune.py`` and ``predict_full_chromosome.py`` accept a ``--compile`` flag:

.. code-block:: bash

   python scripts/predict_full_chromosome.py \
       --model model.pth --fasta hg38.fa --output predictions/ \
       --head atac --compile

This is especially effective for chromosome-scale prediction where 
the large number of batches amortises the one-time compilation cost.


Mixed Precision
---------------

Mixed precision uses bfloat16 for compute while keeping parameters in float32.
This roughly halves GPU memory for activations and speeds up matmuls on Ampere+
GPUs (A100, RTX 30xx and newer).

.. code-block:: python

   from alphagenome_pytorch import AlphaGenome
   from alphagenome_pytorch.config import DtypePolicy

   model = AlphaGenome.from_pretrained(
       'model.pth',
       dtype_policy=DtypePolicy.mixed_precision(),
       device='cuda',
   )

The ``predict()`` method handles autocast automatically. For the CLI scripts:

.. code-block:: bash

   # Inference
   python scripts/predict_full_chromosome.py \
       --model model.pth --fasta hg38.fa --output predictions/ \
       --head atac --dtype-policy mixed_precision

   # Training
   python scripts/finetune.py --amp ...

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Policy
     - Description
   * - ``DtypePolicy.full_float32()``
     - Full float32 (default, works everywhere)
   * - ``DtypePolicy.mixed_precision()``
     - Float32 params, bfloat16 compute (requires Ampere+ GPU)


Resolution Selection
--------------------

The 1bp decoder is the most expensive part of the model. If you only need
128bp-resolution outputs, skip it entirely:

.. code-block:: python

   # Inference — skip the decoder
   outputs = model.predict(
       dna_onehot, organism_idx=0,
       resolutions=(128,),
   )

.. code-block:: bash

   # Full-chromosome prediction at 128bp (default)
   python scripts/predict_full_chromosome.py \
       --model model.pth --fasta hg38.fa --output predictions/ \
       --head atac --resolution 128

   # Finetuning at 128bp only
   python scripts/finetune.py --resolutions 128 ...

Heads that only support 128bp (``chip_tf``, ``chip_histone``) always skip the
decoder regardless of this setting.


Head Selection
--------------

By default, ``forward()`` runs all prediction heads. If you only need a subset,
pass the ``heads`` argument to skip the rest:

.. code-block:: python

   outputs = model.predict(
       dna_onehot, organism_idx=0,
       heads=('atac', 'dnase'),
   )


Gradient Checkpointing
----------------------

Gradient checkpointing trades compute for memory during training by
recomputing activations during the backward pass instead of storing them.

.. code-block:: python

   model = AlphaGenome(gradient_checkpointing=True)

   # Or toggle dynamically
   model.set_gradient_checkpointing(True)

This is a training-only optimization — it has no effect during inference with
``torch.no_grad()``.


Batch Size
----------

Larger batch sizes improve GPU utilisation, especially for chromosome-scale
prediction where the model runs many windows:

.. code-block:: bash

   python scripts/predict_full_chromosome.py \
       --model model.pth --fasta hg38.fa --output predictions/ \
       --head atac --batch-size 8

If you hit out-of-memory errors, reduce batch size or combine with mixed
precision and resolution selection.


Combining Optimizations
-----------------------

These techniques stack. For maximum inference throughput:

.. code-block:: bash

   python scripts/predict_full_chromosome.py \
       --model model.pth --fasta hg38.fa --output predictions/ \
       --head atac \
       --compile \
       --dtype-policy mixed_precision \
       --resolution 128 \
       --batch-size 8

Or equivalently in Python:

.. code-block:: python

   import torch
   from alphagenome_pytorch import AlphaGenome
   from alphagenome_pytorch.config import DtypePolicy

   model = AlphaGenome.from_pretrained(
       'model.pth',
       dtype_policy=DtypePolicy.mixed_precision(),
       device='cuda',
   )
   model.eval()
   model = torch.compile(model)

   outputs = model.predict(
       dna_onehot, organism_idx=0,
       resolutions=(128,),
       heads=('atac',),
   )

Python API
==========

Python API can be used directly. Key functions:

- :func:`~alphagenome_pytorch.extensions.finetuning.transfer.load_trunk` — load pretrained trunk weights (excluding heads)
- :func:`~alphagenome_pytorch.extensions.finetuning.transfer.add_head` — add a new prediction head to the model
- :func:`~alphagenome_pytorch.extensions.finetuning.transfer.remove_all_heads` — remove all existing heads
- :func:`~alphagenome_pytorch.extensions.finetuning.transfer.prepare_for_transfer` — one-step setup: removes/adds heads and applies adapters based on a :class:`~alphagenome_pytorch.extensions.finetuning.transfer.TransferConfig`
- :func:`~alphagenome_pytorch.extensions.finetuning.transfer.count_trainable_params` — count trainable parameters by component
- :func:`~alphagenome_pytorch.extensions.finetuning.adapters.get_adapter_params` — get adapter parameters for optimizer param groups
- :func:`~alphagenome_pytorch.extensions.finetuning.adapters.merge_adapters` — fold adapter weights into base layers for inference

Loading Pretrained Weights
--------------------------

Use :func:`~alphagenome_pytorch.extensions.finetuning.transfer.load_trunk` to load only the trunk weights:

.. code-block:: python

   from alphagenome_pytorch import AlphaGenome
   from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk

   # Create model
   model = AlphaGenome()

   # Load pretrained trunk, excluding heads
   model = load_trunk(model, 'alphagenome_pretrained.pt', exclude_heads=True)

The ``exclude_heads`` parameter (default ``True``) skips loading head weights.

Adding or Removing Heads
------------------------

Use :func:`~alphagenome_pytorch.extensions.finetuning.transfer.remove_all_heads` to strip all pretrained
heads, and :func:`~alphagenome_pytorch.extensions.finetuning.transfer.add_head` to register new ones:

.. code-block:: python

   from alphagenome_pytorch.extensions.finetuning.transfer import remove_all_heads, add_head
   from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head

   # Remove all pretrained heads
   model = remove_all_heads(model)

   # Create and add a new head
   head = create_finetuning_head('atac', n_tracks=4, resolutions=[1, 128])
   add_head(model, 'my_atac', head)

Alternatively, :func:`~alphagenome_pytorch.extensions.finetuning.transfer.prepare_for_transfer` can
handle head removal, creation, and adapter setup in a single call (see
:ref:`managing-heads-transferconfig` below).

.. _managing-heads-transferconfig:

Managing Heads with TransferConfig
-----------------------------------

Configure which heads to remove and add via :class:`~alphagenome_pytorch.extensions.finetuning.transfer.TransferConfig`:

.. code-block:: python

   from alphagenome_pytorch.extensions.finetuning.transfer import TransferConfig, prepare_for_transfer

   config = TransferConfig(
       mode='lora',  # or a list: ['lora', 'locon']
       lora_rank=8,
       lora_alpha=16,
       # Remove original heads you don't need
       remove_heads=['atac', 'dnase', 'chip_tf', 'chip_histone'],
       # Or, alternatively, specify which to keep
       # keep_heads=['rna_seq'],

       # Add new heads for your tracks
       new_heads={
           'my_atac': {
               'modality': 'atac',
               'num_tracks': 10,         # Number of tracks to predict
               'resolutions': [1, 128],  # bp resolutions
           },
       },
   )

   model = prepare_for_transfer(model, config)

Each new head predicts at the specified resolutions. The output will be accessible
as ``outputs['my_atac'][1]`` and ``outputs['my_atac'][128]``.

Complete Example
----------------

End-to-end finetuning with LoRA adapters:

.. code-block:: python

   import torch
   from alphagenome_pytorch import AlphaGenome
   from alphagenome_pytorch.extensions.finetuning.transfer import (
       TransferConfig, load_trunk, prepare_for_transfer, count_trainable_params
   )
   from alphagenome_pytorch.extensions.finetuning.adapters import get_adapter_params, merge_adapters

   # 1. Create model with gradient checkpointing
   model = AlphaGenome(gradient_checkpointing=True)

   # 2. Load pretrained trunk
   model = load_trunk(model, 'alphagenome_pretrained.pt', exclude_heads=True)

   # 3. Configure transfer learning
   config = TransferConfig(
       mode='lora',  # or combine: ['lora', 'locon']
       lora_rank=8,
       lora_alpha=16,
       lora_targets=['q_proj', 'v_proj'],  # Apply LoRA to attention
       remove_heads=['atac', 'dnase', 'chip_tf', 'chip_histone'],
       new_heads={
           'my_atac': {'modality': 'atac', 'num_tracks': 10, 'resolutions': [1, 128]},
       },
   )

   # 4. Prepare model for transfer
   model = prepare_for_transfer(model, config)

   # Check trainable params
   params = count_trainable_params(model)
   print(f"Trainable: {params['total']:,} (heads: {params['heads']:,}, adapters: {params['adapters']:,})")

   # 5. Set up optimizer (only adapter + head params)
   optimizer = torch.optim.AdamW([
       {'params': get_adapter_params(model), 'lr': 1e-4},
       {'params': model.heads.parameters(), 'lr': 1e-3},
   ])

   # 6. Training loop
   model.train()
   model = model.cuda()

   for batch in dataloader:
       sequences = batch['sequence'].cuda()     # (B, L, 4) one-hot
       organism_idx = batch['organism'].cuda()  # (B,) 0=human, 1=mouse
       targets = batch['targets'].cuda()

       optimizer.zero_grad()
       outputs = model(sequences, organism_idx)

       # Access your custom head outputs
       predictions = outputs['my_atac'][128]  # (B, L/128, 10)
       loss = your_loss_fn(predictions, targets)

       loss.backward()
       optimizer.step()

   # 7. Merge adapters for efficient inference
   model.eval()
   model = merge_adapters(model)  # Folds adapter weights into base layers
   torch.save(model.state_dict(), 'finetuned_model.pt')

Saving and Loading Finetuned Models
------------------------------------

After training, only the adapter and head weights need to be saved (~5-10MB)
instead of the full model (~1GB). There are three API levels for working with
these delta weights, from simplest to most flexible:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Level
     - When to use
     - Key functions
   * - **High-level**
     - Loading a finetuned model for inference
     - :meth:`AlphaGenome.from_delta`
   * - **Mid-level**
     - Custom setup before loading (e.g. inspecting config, modifying heads)
     - :func:`load_delta_config`, :func:`prepare_for_transfer`, :func:`load_delta_weights`
   * - **Low-level**
     - Resuming training (includes optimizer/scheduler state)
     - :func:`save_delta_checkpoint`, :func:`load_delta_checkpoint`

Exporting Delta Weights
^^^^^^^^^^^^^^^^^^^^^^^

Use :func:`~alphagenome_pytorch.extensions.finetuning.export_delta_weights` to
export the adapter and head weights for sharing. Must be called *before*
``merge_adapters``. The exported file embeds the ``TransferConfig``, so
recipients only need the delta file plus the base pretrained weights:

.. code-block:: python

   from alphagenome_pytorch.extensions.finetuning import export_delta_weights

   # After training, export for sharing
   export_delta_weights(model, config, "my_lora.safetensors")

Loading Delta Weights (High-level)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to load a finetuned model — one line, like ``from_pretrained``:

.. code-block:: python

   from alphagenome_pytorch import AlphaGenome

   model = AlphaGenome.from_delta(
       "my_lora.safetensors",           # delta weights (adapters + heads)
       "alphagenome_pretrained.pth",    # base pretrained weights
       device="cuda",
   )

.. note::

   ``from_delta()`` and ``load_delta_weights()`` load files created by
   ``export_delta_weights()``, not checkpoint files created by
   ``save_delta_checkpoint()`` or ``--save-delta``. To resume training from a
   checkpoint, use ``load_delta_checkpoint()`` instead (see below).

Loading Delta Weights (Mid-level)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the step-by-step API when you need to inspect or modify the config before
loading, for example to check what adapters or heads a colleague's delta file
contains:

.. code-block:: python

   from alphagenome_pytorch import AlphaGenome
   from alphagenome_pytorch.extensions.finetuning import (
       load_trunk, load_delta_config, load_delta_weights
   )
   from alphagenome_pytorch.extensions.finetuning.transfer import prepare_for_transfer

   # 1. Inspect the delta file
   config = load_delta_config("colleague_lora.safetensors")
   print(config.mode, config.lora_rank, list(config.new_heads.keys()))

   # 2. Create base model and load pretrained trunk
   model = AlphaGenome()
   model = load_trunk(model, "alphagenome_pretrained.pth", exclude_heads=True)

   # 3. Apply config to set up adapters and heads
   model = prepare_for_transfer(model, config)

   # 4. Load the delta weights
   load_delta_weights(model, "colleague_lora.safetensors")

Delta Checkpoints (Low-level)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Delta checkpoints include the full training state (optimizer, scheduler,
metadata) for resuming training. Use these during training, and
``export_delta_weights`` when you're ready to share the final result.

**Saving** (must be done *before* ``merge_adapters``):

.. code-block:: python

   from alphagenome_pytorch.extensions.finetuning import save_delta_checkpoint

   save_delta_checkpoint(
       'my_model.delta.pth',
       model,
       config,
       epoch=10,
       val_loss=0.05,
       track_names=['ATAC_GM12878', 'ATAC_K562'],
   )

**Loading** (to resume training):

.. code-block:: python

   from alphagenome_pytorch import AlphaGenome
   from alphagenome_pytorch.extensions.finetuning import (
       load_trunk, load_delta_checkpoint
   )

   model = AlphaGenome()
   model = load_trunk(model, 'alphagenome_pretrained.pth', exclude_heads=True)

   config, metadata = load_delta_checkpoint('my_model.delta.pth', model)
   print(f"Resuming from epoch {metadata.get('epoch')}")

.. note::

   Delta checkpoints include a hash of the base model structure. If you try to
   load a delta checkpoint with incompatible base weights, you'll get an error.
   Use ``verify_hash=False`` to skip this check if needed.

Exporting Full Model Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After merging adapters, you can export the full model weights (no longer a delta):

.. code-block:: python

   from alphagenome_pytorch.extensions.finetuning import (
       export_model_weights, merge_adapters
   )

   model = merge_adapters(model)
   export_model_weights(model, "finetuned_model.safetensors")

Extracting Embeddings for Custom Heads
--------------------------------------

For maximum flexibility, use ``model.encode()`` to extract embeddings without running
any prediction heads. This is useful when building fully custom architectures:

.. code-block:: python

   # Freeze the backbone
   for param in model.parameters():
       param.requires_grad = False

   # Get embeddings (128bp only for efficiency)
   emb = model.encode(sequences, organism_idx, resolutions=(128,))
   emb_128bp = emb['embeddings_128bp']  # (B, L//128, 3072)

   # For Conv1d heads, use NCL format (avoids transpose)
   emb = model.encode(sequences, organism_idx, resolutions=(128,), channels_last=False)
   emb_128bp = emb['embeddings_128bp']  # (B, 3072, L//128)

   # Pass to your custom head
   predictions = my_custom_head(emb_128bp)

Available embeddings:

- ``embeddings_1bp``: (B, L, 1536) — only if ``1`` in resolutions
- ``embeddings_128bp``: (B, L//128, 3072) — always computed
- ``embeddings_pair``: (B, L//2048, L//2048, 128) — for contact maps

Memory Optimization
-------------------

AlphaGenome requires significant GPU memory. Two key optimizations help:

**Gradient Checkpointing** (less memory but slower):

.. code-block:: python

   model = AlphaGenome(gradient_checkpointing=True)

   # Or enable/disable dynamically
   model.set_gradient_checkpointing(True)

**Mixed Precision** (bfloat16 compute):

.. code-block:: python

   from alphagenome_pytorch.config import DtypePolicy

   # Mixed precision with bfloat16
   model = AlphaGenome(dtype_policy=DtypePolicy.mixed_precision())

**Resolution Selection** (128bp saves memory vs 1bp):

.. code-block:: bash

   # Use only 128bp resolution
   python scripts/finetune.py --resolutions 128 ...


Loss Function
-------------

The finetuning uses a multinomial loss with two components:

1. **Positional loss**: Cross-entropy over positions (where is the signal?)
2. **Count loss**: Poisson regression on total counts (how much signal?)

Controlled by ``--positional-weight`` and ``--count-weight`` arguments (default: 5.0 and 1.0).

The loss is computed over segments for memory efficiency. Configure with
``--num-segments`` and ``--min-segment-size``.

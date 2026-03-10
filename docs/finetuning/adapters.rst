Adapters
========

AlphaGenome supports several transfer learning strategies via
:class:`~alphagenome_pytorch.extensions.finetuning.transfer.TransferConfig`.
See `Yuan et al., 2025 <https://www.biorxiv.org/content/10.1101/2025.05.26.656171v2>`_
for more details about using these adapters for sequence-to-function models and 
`<https://github.com/calico/baskerville/blob/main/docs/transfer_human/transfer.md>`_ 
for how such adapters can be used on other models like Borzoi.

Available Modes
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Mode
     - Trainable Params
     - When to Use
   * - ``linear``
     - Heads only
     - Fast baseline
   * - ``lora``
     - Heads + LoRA adapters
     - Extra expressiveness in addition to the linear baseline
   * - ``locon``
     - Heads + Locon adapters
     - Alternative to LoRA, applied to conv layers
   * - ``ia3``
     - Heads + IA3 scaling
     - Minimal added parameters
   * - ``houlsby``
     - Heads + Houlsby bottleneck adapters
     - Classic bottleneck adapters with residual connection
   * - ``full``
     - All weights
     - Maximum expressiveness

Linear Probing
--------------

The simplest approach: freeze the entire pretrained trunk and train only
the newly added heads. This is the fastest mode and a strong baseline.

.. code-block:: python

   config = TransferConfig(
       mode='linear',
       remove_heads=['atac', 'dnase'],
       new_heads={'my_atac': {'modality': 'atac', 'num_tracks': 4}},
   )
   model = prepare_for_transfer(model, config)

No adapter parameters are injected — only head weights are trainable.

LoRA
----

**Low-Rank Adaptation** adds small trainable low-rank matrices to Linear
layers (typically attention projections) while keeping the trunk frozen.
This is the recommended mode for most use cases.

Reference: `LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021) <https://arxiv.org/abs/2106.09685>`_

.. code-block:: python

   config = TransferConfig(
       mode='lora',
       lora_rank=8,          # Rank of the low-rank matrices
       lora_alpha=16,        # Scaling factor (alpha / rank)
       lora_targets=['q_proj', 'v_proj'],  # Target modules by name
       remove_heads=['atac', 'dnase'],
       new_heads={'my_atac': {'modality': 'atac', 'num_tracks': 4}},
   )
   model = prepare_for_transfer(model, config)

**Parameters:**

- ``lora_rank`` — rank of the decomposition (higher = more expressive, more params)
- ``lora_alpha`` — scaling factor; effective scale is ``alpha / rank``
- ``lora_targets`` — list of substrings to match in module names (e.g. ``['q_proj', 'v_proj']``)

After training, LoRA weights can be merged into the base layers for zero-overhead inference:

.. code-block:: python

   from alphagenome_pytorch.extensions.finetuning import merge_adapters
   model = merge_adapters(model)

Locon
-----

**LoRA for Convolutional layers** applies the same low-rank adaptation
to Conv1D layers. Useful for adapting the convolutional tower.

Reference: `Navigating Text-To-Image Customization: From LyCORIS Fine-Tuning to Model Evaluation (Yeh et al., 2023) <https://arxiv.org/pdf/2309.14859>`_

.. code-block:: python

   config = TransferConfig(
       mode='locon',
       locon_rank=4,         # Rank for conv decomposition
       locon_alpha=1,        # Scaling factor
       locon_targets=['conv_tower'],  # Target conv modules
       remove_heads=['atac'],
       new_heads={'my_atac': {'modality': 'atac', 'num_tracks': 4}},
   )
   model = prepare_for_transfer(model, config)

**Parameters:**

- ``locon_rank`` — rank of the decomposition (default: 4)
- ``locon_alpha`` — scaling factor (default: 1)
- ``locon_targets`` — list of substrings to match Conv1D module names

IA3
---

**Infused Adapter by Inhibiting and Amplifying Inner Activations** learns
a multiplicative scaling vector for layer outputs. Extremely
parameter-efficient — only ``output_dim`` parameters per adapted layer.

Reference: `Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning (Liu et al., 2022) <https://arxiv.org/pdf/2205.05638>`_

.. code-block:: python

   config = TransferConfig(
       mode='ia3',
       ia3_targets=['to_k', 'to_v'],  # Output-scaling targets
       ia3_ff_targets=['fc2'],         # Input-scaling targets (feed-forward)
       remove_heads=['atac'],
       new_heads={'my_atac': {'modality': 'atac', 'num_tracks': 4}},
   )
   model = prepare_for_transfer(model, config)

**Parameters:**

- ``ia3_targets`` — modules for output scaling (IA3)
- ``ia3_ff_targets`` — modules for input scaling (IA3_FF, used in feed-forward layers)

Houlsby Adapters
----------------

**Classic bottleneck adapters** insert a down-projection → activation →
up-projection block with a residual connection. Our implementation follows
the `Baskerville <https://github.com/calico/baskerville>`_ TensorFlow reference,
placing adapters at transformer block boundaries.

Reference: `Parameter-Efficient Transfer Learning for NLP (Houlsby et al., 2019) <https://arxiv.org/abs/1902.00751>`_

**Block-Level Placement**

The default placement inserts adapters after each transformer sub-layer
(MHA and MLP), before the residual add:

.. code-block:: python

   config = TransferConfig(
       mode='houlsby',
       houlsby_latent_dim=8,            # Bottleneck dimension
       houlsby_placement='block',       # Baskerville-style (default)
       houlsby_targets=['mha', 'mlp'],  # Adapt both MHA and MLP blocks
       unfreeze_norm=True,              # Unfreeze LayerNorm/RMSBatchNorm (default)
       remove_heads=['atac'],
       new_heads={'my_atac': {'modality': 'atac', 'num_tracks': 4}},
   )
   model = prepare_for_transfer(model, config)

The computation for each transformer block becomes::

    x = x + adapter(mha(x))    # adapter has internal residual
      = x + mha(x) + bottleneck(mha(x))

    x = x + adapter(mlp(x))
      = x + mlp(x) + bottleneck(mlp(x))

**Linear-Level Placement**

You can also wrap individual Linear layers (similar to LoRA targeting):

.. code-block:: python

   config = TransferConfig(
       mode='houlsby',
       houlsby_latent_dim=8,
       houlsby_placement='linear',
       houlsby_targets=['q_proj', 'v_proj'],  # Target specific projections
       remove_heads=['atac'],
       new_heads={'my_atac': {'modality': 'atac', 'num_tracks': 4}},
   )
   model = prepare_for_transfer(model, config)

**Parameters:**

- ``houlsby_latent_dim`` — bottleneck dimension (default: 8)
- ``houlsby_placement`` — where to insert adapters:
   - ``'block'`` (default): Baskerville-style, at transformer block boundaries
   - ``'linear'``: wrap individual Linear layers
- ``houlsby_targets`` — which components to adapt:
   - For ``'block'``: ``['mha', 'mlp']`` (default), ``['mha']``, or ``['mlp']``
   - For ``'linear'``: module name substrings like ``['q_proj', 'v_proj']``
- ``unfreeze_norm`` — whether to unfreeze normalization layers (default: ``True``).
  This matches Baskerville's behavior where LayerNorm parameters are trained
  alongside adapters.


Combining Adapter Modes
------------------------

Adapter modes (``lora``, ``locon``, ``ia3``, ``houlsby``) can be combined by passing a list
to ``mode``. This applies each adapter type simultaneously — for example,
LoRA on attention layers and Locon on convolutional layers:

.. code-block:: python

   config = TransferConfig(
       mode=['lora', 'locon'],
       # LoRA settings (applied to attention)
       lora_rank=8,
       lora_alpha=16,
       lora_targets=['q_proj', 'v_proj'],
       # Locon settings (applied to convolutions)
       locon_rank=4,
       locon_alpha=1,
       locon_targets=['conv_tower'],
       remove_heads=['atac', 'dnase'],
       new_heads={'my_atac': {'modality': 'atac', 'num_tracks': 4}},
   )
   model = prepare_for_transfer(model, config)

Rules:

- ``'full'`` **cannot** be combined with other modes.
- ``'linear'`` can appear alongside adapter modes — the trunk is frozen and
  adapter layers are injected on top.
- Any subset of ``['lora', 'locon', 'ia3', 'houlsby']`` can be combined.

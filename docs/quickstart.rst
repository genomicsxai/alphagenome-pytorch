Quick Start
===========

This guide will help you get started with AlphaGenome PyTorch.

Loading the Model
-----------------

The easiest way to load AlphaGenome is with ``from_pretrained()``, which loads
the model weights from a single checkpoint file:

.. code-block:: python

   from alphagenome_pytorch import AlphaGenome

   model = AlphaGenome.from_pretrained('fold_1_weights.pth', device='cuda')

The checkpoint file contains both the model parameters and track means required for proper output scaling, so you don't
need to load them separately.

Alternatively, you can use the standard PyTorch ``.load_state_dict()``:

.. code-block:: python

   import torch

   # Load state_dict (contains both weights and track_means as buffers)
   state_dict = torch.load('alphagenome_weights.pth', weights_only=True)

   # Initialize the model
   model = AlphaGenome()

   # Load weights and buffers
   model.load_state_dict(state_dict, strict=False)

Preparing Input
---------------

AlphaGenome expects one-hot encoded DNA sequences with shape ``(batch, length, 4)``
where the 4 channels represent A, C, G, T nucleotides in that order.

.. code-block:: python

   from alphagenome_pytorch.extensions.finetuning import (
       sequence_to_onehot,
       onehot_to_sequence,
   )
   import torch

   dna_str = "ACGTTGAC"
   onehot = sequence_to_onehot(dna_str)  # shape (8, 4), dtype uint8

   # Convert to torch tensor for model input
   onehot_tensor = torch.from_numpy(onehot).float()

   # One-hot array back to string
   dna_str = onehot_to_sequence(onehot)  # "ACGTTGAC"

In real-world scenarios you would likely be loading regions from a reference genome FASTA file:

.. code-block:: python

   import torch
   from pyfaidx import Fasta
   from alphagenome_pytorch.extensions.finetuning import sequence_to_onehot

   # Extract a 1MB region
   with Fasta('hg38.fa') as genome:
       sequence = genome['chr22'][35_000_000 : 35_000_000 + 2**20]

   # Convert to one-hot and add batch dimension
   onehot = sequence_to_onehot(sequence)  # numpy array (1048576, 4)
   onehot_pt = torch.from_numpy(onehot).float().unsqueeze(0)  # (1, 1048576, 4)
   onehot_pt = onehot_pt.to('cuda')

   print(f"Input shape: {onehot_pt.shape}")

.. note::

   AlphaGenome supports variable input sequence length, e.g. we can use 4,096 bp (4KB) sequences up to 1,048,576 bp (1MB).
   Longer sequences provide more context for accurate predictions but require more GPU memory.

Inference
---------

Use the ``predict()`` convenience method for inference:

.. code-block:: python

   organism_idx = 0  # 0 = human, 1 = mouse

   outputs = model.predict(onehot_pt, organism_idx)

   print(f"Available outputs: {list(outputs.keys())}")

It will return outputs in float32.

For more control, you can call the model directly with ``torch.no_grad()``:

.. code-block:: python

   organism_index = torch.tensor([0], dtype=torch.long, device=onehot_pt.device)

   with torch.no_grad():
       outputs = model(onehot_pt, organism_index)

In addition to the sequence itself, the model's `.forward()` requires an organism index and uses `0` for human and `1` for mouse.

Extracting Embeddings
^^^^^^^^^^^^^^^^^^^^^

For fine-tuning or custom heads, use ``model.encode()`` to extract embeddings without
running the prediction heads:

.. code-block:: python

   # Get embeddings only (no head computation)
   emb = model.encode(dna_onehot, organism_idx)

   emb_1bp = emb['embeddings_1bp']      # (batch, seq_len, 1536)
   emb_128bp = emb['embeddings_128bp']  # (batch, seq_len // 128, 3072)
   emb_pair = emb['embeddings_pair']    # (batch, seq_len // 2048, seq_len // 2048, 128)

   # Skip 1bp decoder for efficiency (128bp only)
   emb = model.encode(dna_onehot, organism_idx, resolutions=(128,))

Alternatively, to get embeddings alongside predictions, pass ``return_embeddings=True``:

.. code-block:: python

   outputs = model.predict(dna_onehot, organism_idx, return_embeddings=True)

   emb_1bp = outputs['embeddings_1bp']      # (batch, seq_len, 1536)
   emb_128bp = outputs['embeddings_128bp']  # (batch, seq_len // 128, 3072)

Understanding Outputs
---------------------

The model returns a dictionary with predictions for various genomic assays.
Each output type has predictions at one or more resolutions (1bp and/or 128bp):

.. code-block:: python

   # Available output types
   print(outputs.keys())
   # dict_keys(['atac', 'dnase', 'procap', 'cage', 'rna_seq',
   #            'chip_tf', 'chip_histone', 'contact_maps'])

   # Each output has predictions at different resolutions
   atac_1bp = outputs['atac'][1]      # 1bp resolution
   atac_128bp = outputs['atac'][128]  # 128bp resolution

   # Shape: (batch, sequence_length / resolution, num_tracks)
   print(f"ATAC 1bp shape: {atac_1bp.shape}")
   print(f"ATAC 128bp shape: {atac_128bp.shape}")

Named Outputs (Metadata-Aware Filtering)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use named outputs to filter tracks by metadata (ontology, tissue,
assay, strand, etc.) while keeping tensors and metadata in sync:

.. code-block:: python

   from alphagenome_pytorch.named_outputs import TrackMetadataCatalog
   catalog = TrackMetadataCatalog.load_builtin()
   model.set_track_metadata_catalog(catalog)

   out = model.predict(
       dna_onehot,
       organism_index=0,  # human
       named_outputs=True,
   )

   # Query by ontology directly (no manual mask construction)
   liver_rna = out.rna_seq[1].select(ontology_curie="UBERON:0002107")
   liver_tensor = liver_rna.tensor
   liver_track_names = [track.track_name for track in liver_rna.tracks]

   # Access track metadata fields directly
   for track in liver_rna.tracks:
       print(track.biosample_name)        # Direct attribute access
       print(track.get('assay_title'))    # Safe access with default

   # Filter by multiple criteria, including null checks
   ctcf_tracks = out.chip_tf[128].select(
       transcription_factor='CTCF',
       genetically_modified=None,  # Only unmodified samples
   )

Padding tracks are stripped by default — for example, ATAC returns 167 real
human tracks instead of the 256 raw channels. Pass ``include_padding=True`` to
keep them (useful for training with loss masking).

.. list-table:: Output Types
   :header-rows: 1
   :widths: 18 10 12 12 12

   * - Output
     - Resolutions
     - Human tracks
     - Mouse tracks
     - Raw (incl. padding)
   * - ``atac``
     - 1bp, 128bp
     - 167
     - 18
     - 256
   * - ``dnase``
     - 1bp, 128bp
     - 305
     - 67
     - 384
   * - ``procap``
     - 1bp, 128bp
     - 12
     - ---
     - 128
   * - ``cage``
     - 1bp, 128bp
     - 546
     - 188
     - 640
   * - ``rna_seq``
     - 1bp, 128bp
     - 667
     - 173
     - 768
   * - ``chip_tf``
     - 128bp
     - 1617
     - 127
     - 1664
   * - ``chip_histone``
     - 128bp
     - 1116
     - 183
     - 1152
   * - ``contact_maps``
     - 128bp
     - 28
     - 8
     - 28

Track counts show real (non-padding) tracks returned by ``named_outputs=True``.
The "Raw" column is the full tensor dimension. Both organisms share the same
raw dimensions — padding fills the gap.

See :doc:`named_outputs` for the full guide — more filtering examples, strand
handling, variant scoring, and padding details.

GPU Inference
-------------

For faster inference, ensure the model and inputs are on GPU:

.. code-block:: python

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   model = AlphaGenome.from_pretrained('alphagenome_weights.pth', device=device)
   dna_onehot = dna_onehot.to(device)

   outputs = model.predict(dna_onehot, organism_idx=0)

Mixed Precision
---------------

By default, ``from_pretrained()`` loads the model in float32. For reduced memory
usage and faster inference, use mixed precision with bfloat16 compute:

.. code-block:: python

   from alphagenome_pytorch import AlphaGenome
   from alphagenome_pytorch.config import DtypePolicy

   # Mixed precision: float32 params, bfloat16 compute
   model = AlphaGenome.from_pretrained(
       'alphagenome_weights.pth',
       dtype_policy=DtypePolicy.mixed_precision(),
       device='cuda',
   )

   # predict() automatically handles dtype casting
   outputs = model.predict(dna_onehot, organism_idx=0)

.. list-table:: Precision Options
   :header-rows: 1
   :widths: 30 70

   * - Policy
     - Description
   * - ``DtypePolicy.full_float32()``
     - Full float32 (default, maximum numerical stability)
   * - ``DtypePolicy.mixed_precision()``
     - Float32 params with bfloat16 compute

Next Steps
----------

- :doc:`full_chromosome_prediction` - Genome-wide predictions as BigWig files
- :doc:`finetuning` - Transfer learning on your own genomic tracks
- :doc:`api/model` - Full API reference

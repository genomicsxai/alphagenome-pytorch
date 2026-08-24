Checkpoints, exports, and adapter bundles
==========================================

Fine-tuning produces several different kinds of file, and they are not
interchangeable. This page answers two questions: **what is this file?** and
**how do I load it?**

.. important::

   **Some fine-tuned models are one file. Some are two.**

   *Self-contained* artifacts hold the entire model:

   .. code-block:: bash

      agt predict --checkpoint best_model.pth --head atac ...

   *Delta-shaped* artifacts hold only the **difference** from a base model, so
   they need the base weights as well:

   .. code-block:: bash

      agt predict --model model.pth --checkpoint best_model.delta.pth --head atac ...

   The trade-off is size against self-sufficiency: a self-contained model is
   about 1GB, while a delta is 1-10MB but is useless without the matching base
   weights. If you pass a delta-shaped file on its own, the error tells you so
   and, when the file recorded it, names the base checkpoint you need.

Which file do I have?
---------------------

If you are not sure, ask:

.. code-block:: bash

   agt info best_model.delta.pth

.. list-table::
   :header-rows: 1
   :widths: 22 12 12 18 18 18

   * - Artifact
     - Size
     - Needs base weights?
     - Produced by
     - Load (CLI)
     - Load (Python)
   * - **Full checkpoint** ``best_model.pth``
     - ~1GB
     - No
     - a default training run
     - ``agt predict --checkpoint``
     - ``load_finetuned_model``
   * - **Delta checkpoint** ``best_model.delta.pth``
     - 1-10MB
     - Yes
     - ``--save-delta``
     - ``agt predict --model --checkpoint``
     - ``load_finetuned_model``
   * - **Exported full weights**
     - ~1GB
     - No
     - ``merge_adapters`` + ``export_model_weights``
     - ``agt predict --model``
     - ``AlphaGenome.from_pretrained``
   * - **Exported delta weights** ``.safetensors``
     - 1-10MB
     - Yes
     - ``export_delta_weights``
     - ``agt predict --model --checkpoint``
     - ``load_finetuned_model``
   * - **Adapter bundle** (directory or ``hf://``)
     - 1-10MB
     - Yes
     - ``agt adapters export``
     - ``agt predict --model --checkpoint``
     - ``resolve_checkpoint_and_manifest``

In Python the same classification is available directly, and is what the loader
itself uses:

.. code-block:: python

   from alphagenome_pytorch.extensions.finetuning import describe_checkpoint

   info = describe_checkpoint("best_model.delta.pth")
   info.kind                    # 'delta_checkpoint'
   info.requires_base_weights   # True
   info.base_model_weights_hash # SHA-256 of the base weights, when recorded

Which one should I share?
-------------------------

"What do I have" and "what should I hand someone else" are different questions.

**Share an adapter bundle** unless you have a reason not to. It is
self-describing, records which base model it belongs to so a mismatch fails
loudly instead of silently producing nonsense, publishes to Hugging Face, and is
the only form ``agt serve`` can host straight from a URI.

**Exported delta weights** are a reasonable lighter alternative — a single file,
no manifest — when the recipient already has the right base weights.

**A delta checkpoint** is a *training* artifact, not a distribution format: it
carries optimizer state for resuming a run. It is the input to
``agt adapters export``, not the thing you publish.

**A full checkpoint or full export** is the right choice only when the recipient
cannot obtain the base weights, or when you trained with ``--mode full`` and
there is no adapter to extract. It is ~1GB, but the recipient needs nothing else.

.. note::

   A default training run writes **only full checkpoints**. If you intend to
   share your model, pass ``--save-delta`` — ``agt adapters export`` needs a
   delta checkpoint and will reject ``best_model.pth``. See
   :doc:`cli` for what a run produces.

Full checkpoint
---------------

The default output of a training run: the complete model, plus optimizer state
and the metadata needed to rebuild the heads.

**Produce.** Written automatically — ``best_model.pth`` whenever validation loss
improves, and ``checkpoint_epoch{N}.pth`` each epoch. See
:ref:`what-a-run-produces`.

**Load.**

.. code-block:: bash

   agt predict --checkpoint runs/my_run/best_model.pth --head atac \
       --locus chr1:1000000-1131072 --fasta hg38.fa --output preds/

.. code-block:: python

   from alphagenome_pytorch.extensions.finetuning import load_finetuned_model

   model, meta = load_finetuned_model("runs/my_run/best_model.pth")

Passing the base weights as well is allowed and harmless, but is not required:
the checkpoint already contains every weight, so the base file would simply be
overwritten.

.. dropdown:: Older checkpoints without an embedded config
   :icon: alert

   Checkpoints written before the ``TransferConfig`` was embedded need it
   supplied separately, or the adapter architecture cannot be rebuilt:

   .. code-block:: bash

      agt predict --checkpoint best_model.pth --transfer-config config.json ...

   Newer runs embed the config, so this is only needed for legacy files. If a
   checkpoint contains adapter weights but no config, loading fails with an
   explicit error rather than silently dropping the adapters.

Delta checkpoint
----------------

Only the trained parts — adapters, new heads and trainable norms — plus
optimizer state. One to three orders of magnitude smaller than a full
checkpoint.

**Produce.**

.. code-block:: bash

   agt finetune --mode lora --save-delta ...

This writes ``best_model.delta.pth`` alongside the full checkpoint. Add
``--no-full-checkpoint`` to write *only* deltas. Not available for
``--mode full``, which trains the trunk itself and so has no meaningful delta.

**Load.**

.. code-block:: bash

   agt predict --model model.pth --checkpoint best_model.delta.pth --head atac ...

.. code-block:: python

   model, meta = load_finetuned_model(
       "best_model.delta.pth", pretrained_weights="model.pth",
   )

Exported full weights
---------------------

A plain state dict with adapters folded in — no config, no optimizer state, no
metadata. This is the format to use when the recipient should not need anything
from this project's fine-tuning machinery.

**Produce.**

.. code-block:: python

   from alphagenome_pytorch.extensions.finetuning import (
       merge_adapters, export_model_weights,
   )

   model = merge_adapters(model)          # fold LoRA into the base weights
   export_model_weights(model, "finetuned.safetensors")

**Load.** It is no longer a delta, so it loads exactly like base weights:

.. code-block:: bash

   agt predict --model finetuned.safetensors --head atac ...

.. warning::

   ``export_model_weights`` writes tensors and nothing else — it has no
   parameters for track names, organism or provenance, so there is no way to
   embed them. Record your track names alongside the file, or you will have
   unlabeled output channels.

   Checkpoints written by ``agt finetune`` and bundles built by
   ``agt adapters export`` carry this metadata automatically.
   ``export_delta_weights`` *can* carry it, but only if you pass it — see below.

Exported delta weights
----------------------

The sharing format for a delta: adapters, new heads and trainable norms in a
single ``.safetensors`` file, with the ``TransferConfig`` embedded in the header.

**Produce.** The ``TransferConfig`` is always written, but the *descriptive*
metadata is opt-in — every one of these keyword arguments defaults to ``None``,
and omitting them produces a file whose track names and organism are simply
absent:

.. code-block:: python

   from alphagenome_pytorch.extensions.finetuning import export_delta_weights

   export_delta_weights(
       model, config, "adapter.safetensors",
       track_names={"my_atac": ["K562_rep1", "K562_rep2"]},
       organism="human",
       base_model_weights_hash=base_hash,   # lets loaders detect a wrong base
   )

Passing them matters: they are what ``load_finetuned_model`` returns in ``meta``,
so a recipient of a bare export gets unlabeled output channels. If the delta came
from a training run, ``agt adapters export`` copies this metadata across from the
checkpoint for you — prefer it over calling this function by hand.

**Load.**

.. code-block:: bash

   agt predict --model model.pth --checkpoint adapter.safetensors --head atac ...

.. code-block:: python

   model, meta = load_finetuned_model(
       "adapter.safetensors", pretrained_weights="model.pth",
   )

.. warning::

   ``AlphaGenome.from_delta()`` loads this format too, but **prefer**
   ``load_finetuned_model``.

   ``from_delta`` loads the trunk with ``exclude_heads=True``, so the base
   model's native heads (``atac``, ``dnase``, ``cage``, …) are never populated
   from the pretrained file — they keep the random values the constructor gave
   them — and nothing removes them afterwards. The returned model therefore
   carries randomly-initialised heads alongside your fine-tuned one, so a full
   forward pass or iterating ``model.heads`` yields garbage from every head
   except the one you trained. ``from_delta`` also leaves adapters unmerged.

   ``load_finetuned_model`` strips those heads first and merges adapters by
   default, which is why it is the recommended entry point.

Adapter bundle
--------------

Exported delta weights plus a manifest recording which base model they belong
to. The recommended way to publish a fine-tune. Layout and manifest schema are
documented in :doc:`../serving/adapters`.

**Produce.**

.. code-block:: bash

   agt adapters export \
       --checkpoint runs/my_run/best_model.delta.pth \
       --out dist/k562-atac-lora \
       --id k562-atac-lora \
       --base-weights model.pth

   agt adapters validate dist/k562-atac-lora --base-weights model.pth
   agt adapters publish dist/k562-atac-lora hf://your-org/k562-atac-lora

**Load.** A bundle directory or URI is accepted anywhere a checkpoint is:

.. code-block:: bash

   agt predict --model model.pth --checkpoint dist/k562-atac-lora --head atac ...
   agt predict --model model.pth --checkpoint hf://your-org/k562-atac-lora --head atac ...
   agt serve --weights model.pth --checkpoint hf://your-org/k562-atac-lora

Because the manifest records the base model, loading a bundle against the wrong
base weights **fails** rather than producing quiet nonsense.

.. code-block:: python

   from alphagenome_pytorch.extensions.serving.bundle import (
       resolve_checkpoint_and_manifest,
   )

   path, manifest = resolve_checkpoint_and_manifest("hf://your-org/k562-atac-lora")
   model, meta = load_finetuned_model(path, pretrained_weights="model.pth")

What did I just load?
---------------------

``load_finetuned_model`` returns the model *and* the metadata describing it. A
fine-tune's heads are not the base model's heads, so this is how you find out
what your output channels mean:

.. code-block:: python

   model, meta = load_finetuned_model("best_model.delta.pth", "model.pth")

   meta["head_names"]      # heads this fine-tune added, e.g. ['my_atac']
   meta["track_names"]     # per-head track names, in channel order
   meta["track_metadata"]  # richer per-track rows, when the run recorded them
   meta["organism"]        # 'human' or 'mouse'
   meta["epoch"], meta["val_loss"]

Checkpoints from ``agt finetune`` and bundles from ``agt adapters export`` carry
all of this. A hand-rolled ``export_delta_weights`` call carries only what you
passed it, and an ``export_model_weights`` file carries none of it — see the
warnings under `Exported full weights`_ and `Exported delta weights`_.

.. _sharing-walkthrough:

End-to-end: train, publish, predict
------------------------------------

The whole lifecycle, from a fresh fine-tune to someone else predicting with it.

**1. Train, keeping a shareable artifact.** ``--save-delta`` is what makes the
run produce something you can publish:

.. code-block:: bash

   agt finetune --mode lora --save-delta \
       --genome hg38.fa \
       --modality atac --bigwig data/*.bw \
       --train-bed train.bed --val-bed val.bed \
       --pretrained-weights model.pth \
       --output-dir runs --run-name k562-atac

**2. See what it produced.**

.. code-block:: bash

   ls runs/k562-atac/
   # best_model.pth  best_model.delta.pth  checkpoint_epoch1.pth ...
   # config.json  training_log.csv  epoch_log.csv

**3. Package the delta as a bundle.**

.. code-block:: bash

   agt adapters export \
       --checkpoint runs/k562-atac/best_model.delta.pth \
       --out dist/k562-atac \
       --id k562-atac \
       --base-weights model.pth \
       --organism human --modality atac

**4. Check it before sharing.**

.. code-block:: bash

   agt adapters inspect dist/k562-atac
   agt adapters validate dist/k562-atac --base-weights model.pth

**5. Publish.**

.. code-block:: bash

   agt adapters publish dist/k562-atac hf://your-org/k562-atac

**6. Predict with it, from anywhere.** The recipient needs the bundle and the
same base weights:

.. code-block:: bash

   agt predict \
       --model model.pth \
       --checkpoint hf://your-org/k562-atac \
       --head atac \
       --locus chr1:1000000-1131072 \
       --fasta hg38.fa \
       --output preds/

See also
--------

- :doc:`cli` — training flags and what a run writes to disk
- :doc:`python_api` — the transfer API used during training
- :doc:`../serving/adapters` — bundle layout, manifest schema, catalog serving

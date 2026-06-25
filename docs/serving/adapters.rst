Adapter Bundles and Catalog Serving
====================================

Finetuned adapters can be packaged as **bundles** — a directory containing the
exported delta-weights file plus a sidecar manifest — and served either alone
(singleton mode) or alongside other adapters over a shared base trunk
(catalog mode).

Bundle Layout
-------------

A bundle is a directory:

.. code-block:: text

   my-bundle/
     adapter.safetensors          # output of export_delta_weights
     alphagenome_adapter.json     # sidecar manifest (this format)
     README.md                    # generated model card
     metrics.json                 # optional, copy of evaluation output

The ``adapter.safetensors`` file embeds the ``transfer_config`` (and optional
``track_names`` / ``track_metadata``) used to reconstruct the finetuned model
at load time. The manifest is **display/provenance only**; it does not gate
loading. The manifest's ``base_model_hash`` is cross-checked against the live
base at load time and a mismatch raises a clear error.

Manifest Schema (v1)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "schema_version": 1,
     "id": "wtc11-atac-lora",
     "label": "WTC11 ATAC LoRA",
     "base_model_id": "your-org/alphagenome",
     "base_model_hash": "sha256:abc...",
     "alphagenome_pytorch_version": "x.y.z",
     "adapter_summary": {"kind": "lora", "lora_rank": 8, "lora_alpha": 16},
     "genome": "hg38",
     "organism": "human",
     "modality": "atac",
     "biosample": "WTC11",
     "heads": ["atac_wtc11"],
     "metrics_path": "metrics.json",
     "license": "apache-2.0",
     "provenance": {"created_at": "...", "git_commit": "..."},
     "adapter_filename": "adapter.safetensors"
   }

CLI Workflow
------------

Build a bundle from an existing delta checkpoint:

.. code-block:: bash

   agt adapters export \
     --checkpoint runs/wtc11-atac/best.delta.pth \
     --base-model your-org/alphagenome \
     --id wtc11-atac-lora \
     --label "WTC11 ATAC LoRA" \
     --genome hg38 \
     --organism human \
     --modality atac \
     --biosample WTC11 \
     --out dist/wtc11-atac-lora

Inspect or validate before sharing:

.. code-block:: bash

   agt adapters inspect dist/wtc11-atac-lora
   agt adapters validate dist/wtc11-atac-lora --base-weights model.pth

Pull a bundle (local or Hugging Face) and print its resolved local path:

.. code-block:: bash

   agt adapters pull hf://your-org/alphagenome-wtc11-atac-lora

Publish a bundle to the Hugging Face Hub (requires the ``hf`` extra,
``pip install 'alphagenome-pytorch[hf]'``):

.. code-block:: bash

   agt adapters publish dist/wtc11-atac-lora hf://your-org/alphagenome-wtc11-atac-lora

URI Forms
~~~~~~~~~

Anywhere ``--checkpoint`` or ``--source`` accepts a bundle reference,
the following are recognized:

- bare path or ``local:/abs/path`` — local directory or local file.
- ``file:///abs/path`` — local URL.
- ``hf://org/repo[/subdir][@revision]`` — Hugging Face Hub.

Singleton Serving
-----------------

The existing ``--checkpoint`` flag accepts a delta checkpoint, a delta-weights
``.safetensors`` file, *or* a bundle URI:

.. code-block:: bash

   agt serve \
     --weights base.safetensors \
     --checkpoint hf://your-org/alphagenome-wtc11-atac-lora \
     --fasta hg38.fa \
     --rest-port 8080

Singleton mode is fully compatible with the official ``alphagenome`` gRPC
client because the server has exactly one model and the client does not need
to select one.

Catalog Serving
---------------

Catalog mode hosts multiple finetunes over a shared base trunk and is
**REST-first**. gRPC is supported only for custom clients that pass an
explicit ``alphagenome-model-id`` metadata header.

Catalog file (``adapters.yaml``):

.. code-block:: yaml

   base:
     id: alphagenome-base               # optional; if present, base is also served
     label: AlphaGenome (base)
   adapters:
     - id: wtc11-atac-lora
       source: hf://your-org/alphagenome-wtc11-atac-lora
     - id: k562-rna-locon
       source: local:/srv/bundles/k562-rna-locon

Start the server:

.. code-block:: bash

   agt serve \
     --weights base.safetensors \
     --adapter-catalog adapters.yaml \
     --fasta hg38.fa \
     --rest-port 8080

REST Endpoints
~~~~~~~~~~~~~~

Catalog mode adds two GET routes alongside the existing prediction/scoring
routes:

- ``GET /v1/models`` — list all served models.
- ``GET /v1/models/{id}/metadata`` — output metadata for a specific model.

All POST routes accept an optional top-level ``model_id`` field to route the
request to a specific entry. Scoped variants
(``POST /v1/models/{id}/predict_interval``) are also available.

Model selection rules:

- Singleton mode: ``model_id`` is ignored if absent; rejected with HTTP 400
  if present and does not match.
- Catalog mode with a single model registered: ``model_id`` is optional.
- Catalog mode with multiple models registered: missing ``model_id`` →
  HTTP 400 (ambiguous); unknown ``model_id`` → HTTP 404.

gRPC Behavior
~~~~~~~~~~~~~

Catalog-mode gRPC requires every RPC to carry an ``alphagenome-model-id``
metadata header. Missing → ``FAILED_PRECONDITION``; unknown id →
``NOT_FOUND``. There is no default-model fallback: this is intentional, so
the official ``alphagenome`` client cannot silently bind to the wrong model.
Users who want official-client compatibility should run a singleton process.

How Catalog Mode Works
----------------------

The router holds one base model resident on the device and a list of
``ServedModelEntry`` objects. Each entry captures the finetune's adapter
wrapper modules (``LoRA`` / ``Locon`` / ``IA3`` / ``Houlsby``) and any new
heads. Wrappers store references to the trunk's ``Linear`` / ``Conv1d``
parameters, so all entries share the same base weights.

A request for ``model_id`` triggers a serialized swap: the active entry's
wrappers are detached (``setattr`` back to ``original_layer``), the requested
entry's wrappers are reattached, and any new heads are placed onto
``base_model.heads``. Swap latency is bounded by adapter + head state-dict
size, not by reloading base weights.

Constraints (v1)
~~~~~~~~~~~~~~~~

- Adapters must remain unmerged — pass ``--no-merge-adapters`` semantics
  permanently in catalog mode.
- Do not run with ``torch.compile``; swap mutates the live module tree.
- Bundles whose ``transfer_config`` uses ``keep_heads`` or ``remove_heads``
  cannot be served in catalog mode (the router cannot reversibly remove
  base heads). Use singleton mode for such adapters.

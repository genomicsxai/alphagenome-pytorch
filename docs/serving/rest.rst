REST Serving
============

This page documents the REST (HTTP + JSON) transport exposed by the serving
extension.

Start REST Server
-----------------

Run serving with a REST port:

.. code-block:: bash

   cd /path/to/alphagenome-torch
   source .venv/bin/activate
   agt serve \
     --weights /ABS/PATH/model.pth \
     --fasta /ABS/PATH/hg38.fa \
     --track-metadata /ABS/PATH/track_metadata.parquet \
     --device cuda \
     --host 127.0.0.1 \
     --rest-port 8080

You can run gRPC and REST together by specifying both ``--grpc-port`` and
``--rest-port``.

Remote Access (SSH Tunnel)
--------------------------

If serving runs remotely, tunnel the REST port:

.. code-block:: bash

   ssh -N -L 8080:127.0.0.1:8080 your_user@your_remote_host

Endpoints
---------

- ``POST /v1/predict_sequence``
- ``POST /v1/predict_interval``
- ``POST /v1/predict_variant``
- ``POST /v1/score_variant``
- ``POST /v1/score_variants``
- ``POST /v1/score_ism_variants``
- ``POST /v1/explain_interval``
- ``GET /v1/output_metadata?organism=HOMO_SAPIENS``

Example ``POST`` Request
------------------------

.. code-block:: python

   import json
   import urllib.request

   payload = {
       "sequence": "GATTACA".center(16384, "N"),
       "organism": "HOMO_SAPIENS",
       "requested_outputs": ["DNASE"],
       "ontology_terms": ["UBERON:0002048"],
   }

   req = urllib.request.Request(
       "http://127.0.0.1:8080/v1/predict_sequence",
       data=json.dumps(payload).encode("utf-8"),
       headers={"Content-Type": "application/json"},
       method="POST",
   )

   with urllib.request.urlopen(req, timeout=300) as resp:
       data = json.loads(resp.read().decode("utf-8"))

   values = data["output"]["dnase"]["values"]
   print("rows:", len(values))
   print("cols:", len(values[0]) if values else 0)
   print("metadata row 0:", data["output"]["dnase"]["metadata"][0])

Example ``GET`` Request
-----------------------

.. code-block:: bash

   curl -s "http://127.0.0.1:8080/v1/output_metadata?organism=HOMO_SAPIENS"

``/v1/explain_interval`` — Nucleotide Attribution
--------------------------------------------------

This endpoint computes per-nucleotide attribution scores for a target window
inside a genomic interval. Two methods are supported:

- **input_x_gradient** — gradient × input projection (one forward + backward
  per track; always runs in fp32).
- **saturation_ism** — saturation in-silico mutagenesis (batched forward passes
  mutating every position to every alternate base).

Request fields
~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 10 65
   :header-rows: 1

   * - Field
     - Required
     - Description
   * - ``interval``
     - ✓
     - Full input interval (must be a supported sequence length: 16 384,
       131 072, 524 288, or 1 048 576 bp). Object with ``chromosome``,
       ``start``, ``end``.
   * - ``target_interval``
     - ✓
     - Sub-interval to attribute over — must be contained within
       ``interval``. Same object shape.
   * - ``requested_output``
     - ✓
     - Head name (e.g. ``"dnase"``, ``"atac"``, ``"cage"``).
   * - ``resolution``
     - ✓
     - Output resolution in bp (``1`` or ``128``).
   * - ``track_indices``
     - ✓
     - List of track indices to attribute (e.g. ``[0]``).
   * - ``method``
     - ✓
     - ``"input_x_gradient"`` or ``"saturation_ism"``.
   * - ``organism``
     -
     - Organism identifier (default ``"HOMO_SAPIENS"``).
   * - ``reduction``
     -
     - Window reduction: ``"sum"`` (default), ``"mean"``, or ``"max"``.
       (``"max"`` suits gradient saliency; ``"sum"``/``"mean"`` suit ISM.)
   * - ``include_raw_gradient``
     -
     - If ``true``, include the full ``(W, 4, T)`` gradient tensor. Only
       valid for gradient methods.
   * - ``strand_averaged``
     -
     - If ``true``, average forward and reverse-complement attributions.
   * - ``batch_size``
     -
     - Batch size for the ISM mutation loop (default ``8``).

Response
~~~~~~~~

The response wraps an ``attribution`` object:

.. code-block:: json

   {
     "attribution": {
       "method": "input_x_gradient",
       "kind": "base_matrix",
       "bases": ["A", "C", "G", "T"],
       "values": [[[0.12], [null], [null], [null]], ...],
       "sequence": "AAAC...",
       "target_start": 100,
       "target_end": 200,
       "resolution": 1,
       "track_indices": [0],
       "reduction": "sum",
       "raw_gradient": null,
       "metadata": {"strand_averaged": false}
     }
   }

``values`` has shape ``(W, 4, T)`` where ``W = target_end - target_start``
(in bp), ``4`` is the base axis (A, C, G, T), and ``T`` is the number of
requested tracks.

- For **gradient** methods, only the reference-base cell is filled; all other
  base cells are ``null``.
- For **ISM**, only the mutated cells are filled (the reference-base column is
  ``null``).
- Positions with ``N`` in the reference are all-``null``.

Example: gradient × input
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import json
   import urllib.request

   payload = {
       "interval": {"chromosome": "chr12", "start": 7654368, "end": 7785440},
       "target_interval": {"chromosome": "chr12", "start": 7720000, "end": 7720512},
       "organism": "HOMO_SAPIENS",
       "requested_output": "dnase",
       "resolution": 1,
       "track_indices": [0],
       "method": "input_x_gradient",
       "reduction": "sum",
   }

   req = urllib.request.Request(
       "http://127.0.0.1:8080/v1/explain_interval",
       data=json.dumps(payload).encode("utf-8"),
       headers={"Content-Type": "application/json"},
       method="POST",
   )

   with urllib.request.urlopen(req, timeout=600) as resp:
       data = json.loads(resp.read().decode("utf-8"))

   attr = data["attribution"]
   print("method:", attr["method"])      # input_x_gradient
   print("shape:", len(attr["values"]),   # 512 positions
         len(attr["values"][0]),           # 4 bases
         len(attr["values"][0][0]))        # 1 track

Example: saturation ISM
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   payload = {
       "interval": {"chromosome": "chr12", "start": 7654368, "end": 7785440},
       "target_interval": {"chromosome": "chr12", "start": 7720000, "end": 7720064},
       "requested_output": "dnase",
       "resolution": 1,
       "track_indices": [0],
       "method": "saturation_ism",
       "batch_size": 16,
   }

   # ... same request pattern as above ...

   # ISM reference-base cells are null; mutated cells have deltas
   attr = data["attribution"]
   print("method:", attr["method"])      # saturation_ism


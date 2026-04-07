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

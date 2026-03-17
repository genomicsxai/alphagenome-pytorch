gRPC Serving
============

This page shows how to run the local gRPC server and query it using the
official ``alphagenome`` Python client classes.

Install
-------

Install the package with serving dependencies:

.. code-block:: bash

   cd /path/to/alphagenome-torch
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e ".[serving]"

Start gRPC Server
-----------------

Run the local serving process:

.. code-block:: bash

   cd /path/to/alphagenome-torch
   source .venv/bin/activate
   alphagenome-local-serve \
     --weights /ABS/PATH/model.pth \
     --fasta /ABS/PATH/hg38.fa \
     --track-metadata /ABS/PATH/track_metadata.parquet \
     --device cuda \
     --host 127.0.0.1 \
     --grpc-port 50051

Remote Access (SSH Tunnel)
--------------------------

If the server runs on a remote machine, tunnel the gRPC port:

.. code-block:: bash

   ssh -N -L 50051:127.0.0.1:50051 your_user@your_remote_host

Query with Local ``alphagenome`` Library
----------------------------------------

Use ``DnaClient`` with an insecure channel to your local tunnel:

.. code-block:: python

   import grpc
   from alphagenome.models import dna_client

   channel = grpc.insecure_channel(
       "127.0.0.1:50051",
       options=[
           ("grpc.max_send_message_length", -1),
           ("grpc.max_receive_message_length", -1),
       ],
   )
   client = dna_client.DnaClient(channel=channel)

   sequence = "GATTACA".center(dna_client.SEQUENCE_LENGTH_16KB, "N")
   output = client.predict_sequence(
       sequence=sequence,
       organism=dna_client.Organism.HOMO_SAPIENS,
       requested_outputs=[dna_client.OutputType.DNASE],
       ontology_terms=["UBERON:0002048"],
   )

   print("DNASE values shape:", output.dnase.values.shape)
   print(output.dnase.metadata.head(3).to_string(index=False))

.. note::

   For local serving, instantiate ``dna_client.DnaClient`` directly with your
   channel, as shown above.
   ``dna_client.create(...)`` is intended for the hosted Google endpoint.

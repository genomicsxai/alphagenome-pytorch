Serving
=======

The serving extension exposes a local AlphaGenome-compatible API on top of the
PyTorch model implementation.

It supports two transports:

- gRPC (drop-in for ``alphagenome.models.dna_client.DnaClient``)
- REST (HTTP + JSON endpoints)

.. toctree::
   :maxdepth: 1

   grpc
   rest

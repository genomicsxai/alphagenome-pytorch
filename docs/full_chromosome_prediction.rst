Full Chromosome Prediction
==========================

With AlphaGenome we can generate genome-wide predictions by tiling across entire
chromosomes and stitching results into BigWig files. This comes in handy for
visualising predicted signal tracks in genome browsers.

Command-Line Script
-------------------

The script ``scripts/predict_full_chromosome.py`` wraps the Python API and
writes one BigWig file per chromosome/track.

Quick Start
^^^^^^^^^^^

.. code-block:: bash

   # Predict ATAC track 0 for chr1 at 128bp resolution (default)
   python scripts/predict_full_chromosome.py \
       --model model.pth \
       --fasta hg38.fa \
       --output predictions/ \
       --head atac \
       --tracks 0 \
       --chromosomes chr1

.. code-block:: bash

   # Full genome at 1bp resolution with center cropping
   python scripts/predict_full_chromosome.py \
       --model model.pth \
       --fasta hg38.fa \
       --output predictions/ \
       --head atac \
       --resolution 1 \
       --crop-bp 32768 \
       --batch-size 2

CLI Options
^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Argument
     - Default
     - Description
   * - ``--model``
     - *(required)*
     - Path to model weights (``.pth`` file)
   * - ``--fasta``
     - *(required)*
     - Path to reference genome FASTA file
   * - ``--output``
     - *(required)*
     - Output directory for BigWig files
   * - ``--head``
     - *(required)*
     - Prediction head (``atac``, ``dnase``, ``cage``, ``rna_seq``, ``chip_tf``, ``chip_histone``, ``procap``)
   * - ``--tracks``
     - all
     - Comma-separated track indices to output (e.g. ``0,1,2``)
   * - ``--track-names``
     - ``track_0, …``
     - Comma-separated names for output BigWig files
   * - ``--resolution``
     - ``128``
     - Output resolution in bp (``1`` or ``128``)
   * - ``--crop-bp``
     - ``0``
     - Base pairs to crop from each window edge (e.g. ``32768`` keeps the center ~50%)
   * - ``--batch-size``
     - ``4``
     - Number of windows per inference batch
   * - ``--window-size``
     - ``131072``
     - Model input window size in bp
   * - ``--chromosomes``
     - chr1-22, chrX
     - Comma-separated list of chromosomes to predict
   * - ``--organism``
     - ``0``
     - Organism index (``0`` = human, ``1`` = mouse)
   * - ``--device``
     - ``cuda``
     - PyTorch device
   * - ``--quiet``
     - *off*
     - Suppress progress bars

Python API
----------

The inference extension lives in ``alphagenome_pytorch.extensions.inference``.

Predicting a Single Chromosome
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~alphagenome_pytorch.extensions.inference.predict_full_chromosome`
returns predictions for one chromosome as a NumPy array:

.. code-block:: python

   from alphagenome_pytorch import AlphaGenome
   from alphagenome_pytorch.extensions.inference import (
       TilingConfig,
       predict_full_chromosome,
   )

   model = AlphaGenome.from_pretrained("model.pth", device="cuda")

   config = TilingConfig(resolution=1, batch_size=8)

   preds = predict_full_chromosome(
       model,
       "hg38.fa",
       chrom="chr1",
       head="atac",
       config=config,
   )
   # preds.shape == (chrom_length // resolution, n_tracks)

Writing BigWig Files
^^^^^^^^^^^^^^^^^^^^

:func:`~alphagenome_pytorch.extensions.inference.predict_full_chromosomes_to_bigwig`
predicts multiple chromosomes and saves each as a BigWig:

.. code-block:: python

   from alphagenome_pytorch.extensions.inference import (
       TilingConfig,
       predict_full_chromosomes_to_bigwig,
   )

   config = TilingConfig(resolution=128, crop_bp=32768)

   results = predict_full_chromosomes_to_bigwig(
       model=model,
       fasta_path="hg38.fa",
       output_dir="./predictions",
       head="atac",
       chromosomes=["chr1", "chr2"],
       config=config,
       track_indices=[0, 1],        # optional: subset of tracks
       track_names=["sample_A", "sample_B"],  # optional: BigWig names
   )
   # results == {'chr1': [Path('predictions/atac_chr1_sample_A.bw'), ...], ...}

Tiling Configuration
--------------------

:class:`~alphagenome_pytorch.extensions.inference.TilingConfig` controls how the
genome is split into overlapping windows:

.. code-block:: python

   config = TilingConfig(
       window_size=131_072,  # model input size (default)
       crop_bp=32_768,       # crop edges to reduce artefacts
       resolution=128,       # 128bp bins (faster) or 1 (base-pair)
       batch_size=4,         # windows per batch
   )

.. list-table:: TilingConfig fields
   :header-rows: 1
   :widths: 18 12 70

   * - Field
     - Default
     - Description
   * - ``window_size``
     - ``131072``
     - Input window size in bp
   * - ``crop_bp``
     - ``0``
     - Base pairs to crop from *each* edge.
       Setting this enables overlapping windows so only the center of each
       window is kept, reducing edge artefacts.
   * - ``resolution``
     - ``128``
     - ``1`` for base-pair resolution (requires decoder, slower) or ``128``
       for bin-level resolution (faster)
   * - ``batch_size``
     - ``4``
     - Number of windows processed per forward pass

Derived properties:

- ``effective_size`` — kept region per window: ``window_size - 2 * crop_bp``
- ``step_size`` — equals ``effective_size`` for seamless tiling

.. tip::

   Setting ``crop_bp=32768`` (25% of the default 131 072 bp window) keeps the
   central ~50% of each window. This is a good starting point for reducing
   edge prediction artefacts.

Supported Heads
---------------

.. list-table::
   :header-rows: 1
   :widths: 18 12 12

   * - Head
     - Tracks
     - Resolutions
   * - ``atac``
     - 256
     - 1, 128
   * - ``dnase``
     - 384
     - 1, 128
   * - ``procap``
     - 128
     - 1, 128
   * - ``cage``
     - 640
     - 1, 128
   * - ``rna_seq``
     - 768
     - 1, 128
   * - ``chip_tf``
     - 1664
     - 128 only
   * - ``chip_histone``
     - 1152
     - 128 only

.. note::

   ``chip_tf`` and ``chip_histone`` only support 128bp resolution.
   Requesting ``--resolution 1`` with these heads will raise an error.

Performance Tips
----------------

- Use **resolution 128** when 1bp resolution is not needed.
- Use **larger batch size** (``--batch-size 8``) if your GPU memory allows.
- For quick tests, **limit chromosomes** with ``--chromosomes chr21,chr22``.
- Try loading the model with **mixed precision** (``DtypePolicy.mixed_precision()``).

API Reference
-------------

.. autoclass:: alphagenome_pytorch.extensions.inference.TilingConfig
   :members:
   :undoc-members:

.. autofunction:: alphagenome_pytorch.extensions.inference.predict_full_chromosome

.. autofunction:: alphagenome_pytorch.extensions.inference.predict_full_chromosomes_to_bigwig

.. autofunction:: alphagenome_pytorch.extensions.inference.write_bigwig

.. autoclass:: alphagenome_pytorch.extensions.inference.GenomeSequenceProvider
   :members:
   :undoc-members:

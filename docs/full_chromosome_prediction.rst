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
   * - ``--dtype-policy``
     - ``full_float32``
     - Dtype policy for model inference. Use ``full_float32`` for maximum
       compatibility or ``mixed_precision`` for lower GPU memory usage on
       supported hardware
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

.. _per-gene-counts:

Per-Gene Counts and AnnData
---------------------------

Instead of writing BigWig tracks, whole-chromosome predictions can be
**aggregated into a per-gene × per-track matrix** and exported as an AnnData
(``.h5ad``) — a gene expression count table you can hand straight to scanpy.
Tiles are streamed through an accumulator, so a gene whose exons straddle a tile
boundary is still summed correctly and nothing is held in memory beyond the
running ``[gene, track]`` matrix.

This is most useful for the ``rna_seq`` head, but works for any head.

.. note::

   This feature lives on the ``agt predict`` CLI, not on
   ``scripts/predict_full_chromosome.py``. It requires
   ``pip install 'alphagenome-pytorch[inference-anndata]'`` (adds ``pandas``,
   ``anndata``, ``pyranges`` for GTF and ``pyarrow`` for parquet).

Command line
^^^^^^^^^^^^

.. code-block:: bash

   # Per-gene RNA-seq counts for chr20, sense strand only
   agt predict \
       --model model.pth --fasta hg38.fa --output out/ \
       --head rna_seq --chromosomes chr20 \
       --resolution 1 --crop-bp 16384 \
       --anndata gene_counts.h5ad \
       --annotation gencode.v46.parquet \
       --aggregate-over exons --aggregate-func sum --gene-strand match

``--anndata NAME`` replaces the BigWig output and writes ``{output}/NAME``. It
is only valid together with ``--chromosomes`` — the other input modes
(``--locus``, ``--bed``, ``--sequences``) error out, since aggregation is
defined over whole chromosomes. ``--annotation`` is required.

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - Flag
     - Default
     - Description
   * - ``--anndata NAME``
     - *(off)*
     - Write a per-gene × per-track ``.h5ad`` with this filename into
       ``--output``, instead of BigWigs. ``--chromosomes`` mode only.
   * - ``--annotation PATH``
     - *(required)*
     - Gene annotation, GTF/GFF or parquet. Must include exon rows when
       ``--aggregate-over exons``.
   * - ``--aggregate-over``
     - ``exons``
     - Aggregate each gene's signal over its ``exons`` or its full
       ``gene-body`` (exons + introns).
   * - ``--aggregate-func``
     - ``sum``
     - Cell value: ``sum`` (raw counts), ``mean`` (per-base coverage), or
       ``log-mean`` (``log1p`` of the mean).
   * - ``--gene-strand``
     - ``all``
     - ``all`` fills every cell; ``match`` sets strand-incompatible
       (gene, track) cells to NaN.
   * - ``--track-strands``
     - *(from metadata)*
     - Override per-track strands for ``--gene-strand match``: one char per
       output track, compact (``+-+-``) or comma-separated (``+,-,+,-``).
   * - ``--keep-padding``
     - *(off)*
     - Keep placeholder padding tracks in ``obs``. Dropped by default — they
       carry no signal. Only useful for positional parity with raw head channels.

Choosing an aggregation region
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``exons`` (the default)
   Signal over the gene's annotated exonic blocks only. With
   ``--aggregate-func log-mean`` and ``--gene-strand match`` the cell value is
   log-transformed mean exonic coverage, strand-matched — the same quantity
   :func:`~alphagenome_pytorch.aggregation.gene_expression` computes per window.
   It is not identical to that function's output, though: the whole-chromosome
   path has no ``min_exon_fraction`` gene filter (it does not need one, since
   tiling covers every exon), so the *gene set* is every gene in the
   annotation rather than only those sufficiently contained in one window.

``gene-body``
   Exons *plus* introns, using the same gene-body masks as the training gene-LFC
   loss. Use this when you want counts comparable with what the fine-tuning
   pipeline optimizes, or when your annotation has no exon rows.

.. warning::

   **Units.** AlphaGenome's 128bp predictions are *bin sums*, not per-base
   values (``heads.predictions_scaling`` multiplies by the resolution to reach
   experimental space). The ``mean`` / ``log-mean`` reductions divide by
   **bases**, not by elements, so their values are comparable across
   ``--resolution``.

   That buys unit consistency, not exact agreement between resolutions. At
   128bp a bin that an exon only partly covers is an already-summed total
   mixing exonic and non-exonic bases, and no divisor can separate them after
   the fact. Only regions whose boundaries land on bin edges agree exactly with
   the 1bp result. Use ``--resolution 1`` when you need exon boundaries
   respected.

Strand handling
^^^^^^^^^^^^^^^

For stranded assays like RNA-seq, pass ``--gene-strand match``. The default
``all`` also scores every gene with tracks reading the *opposite* strand — that
signal is antisense, not the gene's own expression, and it will quietly inflate
downstream summaries.

``match`` writes NaN into those cells rather than 0, so they drop out of means
and correlations instead of averaging in as zeros. Unstranded (``.``) tracks
match every gene.

Track strands are read from metadata — built in for pretrained heads, taken
from the checkpoint for finetuned ones. ``--track-strands`` is only needed for
custom or legacy heads that carry no strand information.

Preparing the annotation
^^^^^^^^^^^^^^^^^^^^^^^^

No annotation is bundled; bring your own GTF or parquet. Parquet loads in
seconds where a ~1.5 GB GENCODE GTF takes minutes, so convert once:

.. code-block:: bash

   python scripts/convert_gtf_to_parquet.py \
       --input gencode.v46.annotation.gtf \
       --output gencode.v46.parquet

The converter preserves all features, so the resulting parquet works for both
``--aggregate-over exons`` and ``gene-body``. Aggregating over exons against an
annotation with no exon rows is rejected up front with an explicit error.

Output layout
^^^^^^^^^^^^^

The AnnData follows the scanpy convention of one row per *observation*, and
here the observations are tracks:

- ``X`` — ``[n_tracks, n_genes]``
- ``obs`` — one row per track, carrying the full metadata the catalog knows:
  ``track_index``, ``track_name``, ``output_name``, ``organism``, ``strand``,
  ``biosample_name``, ``assay_title``, ``ontology_curie``, and the remaining
  extras. Metadata comes from the built-in catalog for a pretrained model and
  from the embedded rows for a fine-tuned checkpoint. For a custom head with no
  metadata at all, ``obs`` falls back to ``track_index`` only (plus
  ``track_name`` from ``--track-names`` and ``strand`` from ``--track-strands``);
  the CLI prints a note when that happens.
- ``var`` — one row per gene: ``gene_id``, ``gene_name``, ``gene_type``,
  ``strand``, ``Start``, ``End``
- ``obs_names`` are track names (falling back to the track index),
  ``var_names`` are gene ids

Transpose with ``adata.T`` if you want genes as observations.

.. note::

   **Padding tracks are dropped by default.** Head widths are padded to a fixed
   size — the human ``rna_seq`` head is 768 wide but holds only 667 real tracks —
   and the padding channels carry no signal (PyTorch scales them to 0.0), so
   leaving them in dilutes any summary taken across tracks. ``--keep-padding``
   (``strip_padding=False``) keeps them, which is what you want only when ``obs``
   rows must line up positionally with the raw head channels, e.g. for JAX
   parity checks. Identifying padding needs track metadata, so for a custom head
   without it nothing is dropped.

.. important::

   ``var_names`` and ``var["gene_id"]`` carry **version-stripped** gene ids:
   ``ENSG00000141510``, not ``ENSG00000141510.16``. Genes are keyed on the base
   id so that a gene appearing in several tiles accumulates into one row.
   Strip the version on your side too before joining against a GENCODE table,
   or the join silently matches nothing.

.. code-block:: python

   import anndata

   adata = anndata.read_h5ad("out/gene_counts.h5ad")
   adata.shape                    # (n_tracks, n_genes)
   adata.var["gene_name"].head()
   expr = adata.T                 # genes x tracks

Python API
^^^^^^^^^^

:func:`~alphagenome_pytorch.extensions.inference.predict_full_chromosomes_to_anndata`
is what the CLI calls. It returns a
:class:`~alphagenome_pytorch.aggregation.GeneCounts`, and writes the ``.h5ad``
only when ``output_path`` is given:

.. code-block:: python

   from alphagenome_pytorch import AlphaGenome
   from alphagenome_pytorch.extensions.inference import (
       TilingConfig,
       predict_full_chromosomes_to_anndata,
   )

   from alphagenome_pytorch.named_outputs import TrackMetadataCatalog

   model = AlphaGenome.from_pretrained("model.pth", device="cuda")

   # The CLI resolves this for you; a direct caller must pass it. Without it obs
   # holds track_index only and padding tracks cannot be identified or dropped.
   tracks = TrackMetadataCatalog.load_builtin(0).get_tracks(
       "rna_seq", organism=0, strict=True
   )

   gene_counts = predict_full_chromosomes_to_anndata(
       model,
       "hg38.fa",
       "gencode.v46.parquet",
       head="rna_seq",
       chromosomes=["chr20", "chr21"],
       config=TilingConfig(resolution=1, crop_bp=16384),
       track_metadata=tracks,   # rich obs + lets strip_padding find padding
       strip_padding=True,      # default; False keeps raw head channel alignment
       over="exons",            # or "gene_body"
       reduce="sum",            # or "mean"
       log=False,               # log1p after the reduce
       strand="match",          # None / "match" / "merge"
       output_path="gene_counts.h5ad",   # optional
   )

   adata = gene_counts.to_anndata()      # obs=tracks, var=genes
   tidy = gene_counts.to_dataframe()     # long table, one row per (gene, track)

``track_metadata`` must align with ``track_indices`` — pass the full head's
metadata when predicting all tracks, or the matching subset when you narrow with
``track_indices``. Mismatched lengths raise rather than mislabel the rows.

The Python API exposes one mode the CLI does not: ``strand="merge"`` sums
matching ``+``/``-`` track pairs into a single unstranded column, roughly
halving the track axis. ``over`` takes ``"gene_body"`` with an underscore here,
where the CLI flag spells it ``gene-body``.

To aggregate a **single window** rather than whole chromosomes — including the
exon-based gene-expression quantity and the correlation metrics — use
:mod:`alphagenome_pytorch.aggregation` directly; see :doc:`api/aggregation`.

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

.. autofunction:: alphagenome_pytorch.extensions.inference.predict_full_chromosomes_to_anndata

.. autofunction:: alphagenome_pytorch.extensions.inference.write_bigwig

.. autoclass:: alphagenome_pytorch.extensions.inference.GenomeSequenceProvider
   :members:
   :undoc-members:

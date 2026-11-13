Aggregation
===========

``alphagenome_pytorch.aggregation`` turns per-position predictions
(``[B, S, C]``) into per-region matrices — most usefully **per-gene × per-track**
counts and expression values.

.. note::

   These names are **not** re-exported from the package root. Import them from
   the submodule::

      from alphagenome_pytorch.aggregation import gene_expression, aggregate_genes

   That placement is deliberate and pinned by
   ``tests/unit/test_public_api.py``: the core of this module is pure-tensor
   code with no ``pandas`` / ``anndata`` import, so a training loop can use it
   without pulling optional dependencies in.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - What it does
   * - :func:`~alphagenome_pytorch.aggregation.aggregate_intervals`
     - The shared positional primitive: ``[B, S, C]`` × ``[S, R]`` mask →
       ``[B, R, C]``. Pure tensor.
   * - :func:`~alphagenome_pytorch.aggregation.aggregate_genes`
     - Gene-**body** counts (exons + introns), linear space. Mirrors the
       training gene-LFC aggregation.
   * - :func:`~alphagenome_pytorch.aggregation.gene_expression`
     - Gene expression: log mean coverage over annotated **exons**,
       strand-matched.
   * - :class:`~alphagenome_pytorch.aggregation.GeneCounts`
     - Result object: ``counts`` tensor + gene/track metadata, with
       ``.to_dataframe()`` / ``.to_tables()`` / ``.to_anndata()``.
   * - :class:`~alphagenome_pytorch.aggregation.GeneCountAccumulator`
     - Streaming accumulation across tiled windows (used by whole-chromosome
       inference).
   * - :func:`~alphagenome_pytorch.aggregation.gene_expression_values`
     - Single-window ``[G, C]`` expression, for the fine-tuning validation
       metric.
   * - :func:`~alphagenome_pytorch.aggregation.combine_gene_expression`
     - Deduplicate genes across windows and reduce to correlations.
   * - :func:`~alphagenome_pytorch.aggregation.normalize_expression`
     - Quantile-normalize across genes per track, then gene-mean-center.
   * - :func:`~alphagenome_pytorch.aggregation.gene_expression_correlations`
     - The three gene-expression correlation flavors.

Two aggregation definitions
---------------------------

The two user-facing helpers answer different questions, and they are not
interchangeable:

:func:`~alphagenome_pytorch.aggregation.aggregate_genes` — **gene body**
   Signal over exons *and* introns, in linear space, using the same gene-body
   masks as the training gene-LFC loss. Its ``gene_table`` comes from
   :func:`~alphagenome_pytorch.extensions.finetuning.gene_annotation.cached_load_gene_table`
   (parquet or GTF), so this is the quantity to compare against what fine-tuning
   optimizes.

:func:`~alphagenome_pytorch.aggregation.gene_expression` — **exons**
   The AlphaGenome gene-expression quantity: log-transformed mean coverage over
   a gene's annotated exons, strand-matched by default, keeping genes with at
   least ``min_exon_fraction`` (default 50%) of their exons fully inside the
   window. Takes a :class:`~alphagenome_pytorch.variant_scoring.annotations.GeneAnnotation`
   built from a GTF/parquet that includes exon rows.

Both return a :class:`~alphagenome_pytorch.aggregation.GeneCounts`.

.. warning::

   **Units.** 128bp predictions are bin *sums*, not per-base values, so a mean
   over a region must divide by bases rather than elements — that is what the
   ``bin_size`` argument of
   :func:`~alphagenome_pytorch.aggregation.aggregate_intervals` is for, and the
   gene helpers set it from the interval width automatically.

   This gives unit consistency across resolutions, not exact agreement: at
   128bp a bin an exon only partly covers is an already-summed total mixing
   exonic and non-exonic bases. Only regions whose boundaries land on bin edges
   match the 1bp result exactly.

Aggregating one window
----------------------

.. code-block:: python

   import torch
   from alphagenome_pytorch import AlphaGenome
   from alphagenome_pytorch.aggregation import gene_expression
   from alphagenome_pytorch.variant_scoring.annotations import GeneAnnotation

   model = AlphaGenome.from_pretrained("model.pth", device="cuda")
   annotation = GeneAnnotation("gencode.v46.parquet")

   interval = ("chr20", 30_000_000, 30_131_072)   # 0-based half-open

   # Restrict to the head/resolution you need — the default computes every head
   # at both resolutions, which is far more work than this aggregation uses.
   out = model.predict(dna, 0, heads=("rna_seq",), resolutions=(1,))
   preds = out["rna_seq"][1]                       # [B, S, C] at 1bp

   gc = gene_expression(
       preds,
       annotation,
       interval,
       log="log1p",        # or "log" / None for linear
       strand="match",     # NaN out antisense (gene, track) cells
       reduce="mean",
       min_exon_fraction=0.5,
   )

   gc.counts.shape       # [B, n_genes, n_tracks]
   adata = gc.to_anndata()          # obs=tracks, var=genes, X=[tracks, genes]
   tidy = gc.to_dataframe()         # long: one row per (interval, gene, track)

Passing ``track_metadata`` (a sequence of
:class:`~alphagenome_pytorch.named_outputs.TrackMetadata`) labels the track axis
and supplies the strands that ``strand="match"`` needs. Without it the tracks
are labelled by index only, and strand matching has nothing to match on.

``strand`` accepts:

- ``None`` / ``"ignore"`` / ``"all"`` — no strand logic, every cell filled
- ``"match"`` — NaN in strand-incompatible cells (unstranded ``.`` tracks match
  everything)
- ``"merge"`` — sum ``+``/``-`` track pairs that share all other metadata,
  roughly halving the track axis

.. important::

   Gene ids are **version-stripped**: ``gene_metadata["gene_id"]`` and the
   AnnData ``var_names`` hold ``ENSG00000141510``, not
   ``ENSG00000141510.16``. Genes are keyed on the base id so a gene seen in
   several windows or tiles accumulates into one row. Strip the version on your
   side before joining against a GENCODE table.

.. note::

   ``to_tables()`` and ``to_anndata()`` require a single interval (``B == 1``)
   so the matrix layout is unambiguous; they raise otherwise. Index one window
   or loop over the batch. ``to_dataframe()`` handles any ``B``.

Correlation metrics
-------------------

:func:`~alphagenome_pytorch.aggregation.gene_expression_correlations` computes
the three flavors reported by the fine-tuning gene-expression evaluation, given
``[G, C]`` predicted and observed matrices in log space:

``across_genes``
   Raw Pearson over genes, one *r* per track, then averaged.

``across_genes_norm``
   Same, after quantile-normalizing across genes per track and gene-mean-
   centering (:func:`~alphagenome_pytorch.aggregation.normalize_expression`) —
   the specificity-normalized view.

``across_tracks_norm``
   The normalized data correlated per gene across tracks, then averaged.

NaN cells (from strand matching) are handled pairwise for the raw flavor;
the normalized flavors first drop rows/columns that are entirely NaN.

.. code-block:: python

   from alphagenome_pytorch.aggregation import (
       gene_expression_values,
       combine_gene_expression,
   )

   windows = []
   cache = {}      # reuse across windows/epochs: memoizes the pandas lookup
   for interval, pred, obs in val_windows:
       p, gene_ids, _ = gene_expression_values(
           pred, annotation, interval, track_strands=strands, window_cache=cache
       )
       o, _, _ = gene_expression_values(
           obs, annotation, interval, track_strands=strands, window_cache=cache
       )
       windows.append((gene_ids, p, o))

   metrics = combine_gene_expression(windows)
   # {'across_genes': ..., 'across_genes_norm': ...,
   #  'across_tracks_norm': ..., 'n_genes': ...}

``combine_gene_expression`` deduplicates genes by id across windows (first
occurrence wins), so overlapping windows do not double-count. This is exactly
what ``agt finetune --gene-expr-eval`` does each validation epoch — see
:doc:`/finetuning/cli`.

Whole-chromosome aggregation
----------------------------

For tiled whole-chromosome runs, use
:class:`~alphagenome_pytorch.aggregation.GeneCountAccumulator` — feed each tile
with its coordinates and it maintains a running ``[gene, track]`` matrix, so a
gene spanning several tiles is summed across all of them:

.. code-block:: python

   from alphagenome_pytorch.aggregation import GeneCountAccumulator

   acc = GeneCountAccumulator(annotation, resolution=128, over="exons", reduce="sum")
   for tile, chrom, start_bp, end_bp in tiles:
       acc.add_tile(tile, chrom, start_bp, end_bp)

   gc = acc.to_gene_counts(track_metadata=track_frame, log=False, strand="match")

:func:`~alphagenome_pytorch.extensions.inference.predict_full_chromosomes_to_anndata`
wraps this with tiled inference, and ``agt predict --anndata`` wraps that — see
:ref:`per-gene-counts`.

Module Reference
----------------

.. automodule:: alphagenome_pytorch.aggregation
   :members: aggregate_intervals, aggregate_genes, gene_expression,
             gene_expression_values, combine_gene_expression,
             normalize_expression, gene_expression_correlations,
             GeneCounts, GeneCountAccumulator
   :show-inheritance:

Gene annotation
---------------

The exon-based helpers take a ``GeneAnnotation``, which lives in the
variant-scoring package but is documented here because aggregation is its main
consumer. Build it from a GTF/parquet that includes exon rows, or from a
pre-filtered DataFrame.

.. autoclass:: alphagenome_pytorch.variant_scoring.annotations.GeneAnnotation
   :members:
   :show-inheritance:

The gene-**body** helper :func:`~alphagenome_pytorch.aggregation.aggregate_genes`
takes a different object — a plain gene table, the same one the training gene-LFC
loss uses:

.. autofunction:: alphagenome_pytorch.extensions.finetuning.gene_annotation.cached_load_gene_table

.. autoclass:: alphagenome_pytorch.extensions.finetuning.gene_annotation.GeneMaskExtractor
   :members:
   :show-inheritance:

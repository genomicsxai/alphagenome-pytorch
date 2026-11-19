Command-Line Interface
======================

``agt finetune`` is the entry point for fine-tuning using the CLI,
and it can be configured with CLI flags, a YAML config file, or both.

.. note::

   ``agt finetune`` is analogous to running ``python scripts/finetune.py``
   but it works system-wide and ships with the installed package.


For single-process training, use ``--device cpu``, ``--device mps``, or
``--device cuda:N`` to select a device explicitly. The default, ``auto``,
tries CUDA, MPS, then CPU. With ``torchrun``, use ``--device auto`` or
``--device cuda`` so each process uses its assigned CUDA device.

When omitted, ``--dtype`` defaults to ``bfloat16`` on CUDA with AMP enabled and
``float32`` otherwise. An explicit ``--dtype bfloat16`` requires AMP and a
device with bf16 autocast support.


YAML Configuration
------------------

For reproducible experiments, use ``--config config.yaml``.

.. code-block:: bash

   pip install pyyaml
   agt finetune --config config.yaml

.. dropdown:: Full Config Schema
   :icon: code-square

   .. code-block:: yaml

      # =============================================================================
      # Data Configuration
      # =============================================================================

      genome: /path/to/hg38.fa           # Reference genome FASTA (required)
      train_bed: /path/to/train.bed      # Training regions BED file (required)
      val_bed: /path/to/val.bed          # Validation regions BED file (required)
      sequence_length: 131072            # Input sequence length (default: 131072)

      # Global output resolutions - can be overridden per-modality
      # Use "1" for 1bp resolution, "128" for 128bp, or "1,128" for both
      resolutions: "1"                   # String or list: "1", "128", "1,128", or [1, 128]

      # Caching options (memory vs speed tradeoff)
      cache_genome: false                # Cache genome in memory (~12GB for hg38)
      cache_signals: false               # Cache BigWig signals in memory
      max_io_workers: 16                 # Max threads for parallel BigWig I/O

      # =============================================================================
      # Model Configuration
      # =============================================================================

      pretrained_weights: /path/to/model.pth  # Pretrained weights file (required)

      # Training mode: 'linear-probe', 'lora', 'locon', 'lora+locon', or 'full'
      mode: lora

      # LoRA configuration (used when mode includes LoRA)
      lora_rank: 8                       # LoRA rank (0 disables LoRA, trains heads only)
      lora_alpha: 16                     # LoRA alpha scaling factor
      lora_targets: "q_proj,v_proj"      # Comma-separated list of target modules

      # Locon configuration (used when mode includes Locon)
      locon_rank: 4
      locon_alpha: 1
      locon_targets: "down_blocks.4,down_blocks.5"  # Required when Locon is enabled

      device: auto                       # 'auto', 'cpu', 'mps', 'cuda', or 'cuda:N'

      # Model precision
      dtype: null                        # Device default, or 'bfloat16' or 'float32'

      # Head initialization
      head_init_scheme: truncated_normal # 'truncated_normal' or 'uniform'

      # Memory optimization
      gradient_checkpointing: true       # Enable gradient checkpointing

      # =============================================================================
      # Modality Configuration
      # =============================================================================

      # Define one or more modalities with their BigWig files
      modalities:
        atac:                            # Modality name (must be a supported type)
          bigwig:                        # List of BigWig files for this modality
            - /path/to/sample1_atac.bw
            - /path/to/sample2_atac.bw
          resolutions: "1,128"           # Per-modality resolution override (optional)
          task_weight: 1.0               # Loss weight for this modality (optional)

        rna_seq:
          bigwig:
            - /path/to/sample1_rna_plus.bw
            - /path/to/sample1_rna_minus.bw
          resolutions: "128"             # RNA-seq at 128bp only
          task_weight: 0.5               # Lower weight for RNA-seq

          # Per-track strand, one entry per bigwig above, '+' / '-' / '.'.
          # Required when gene_loss_weight > 0. CLI: --track-strands
          strand: "+-"

          # Average +/- track means so paired strands share a scaling factor
          # (recommended for stranded RNA-seq / CAGE / PRO-cap).
          # 'auto' pairs consecutive bigwigs: (0,1), (2,3), ...
          # Explicit form is a list of [plus, minus] index pairs:
          #   strand_pairs: [[0, 1], [2, 3]]
          # CLI: --strand-pairs 'rna_seq:auto'  (overrides this value)
          strand_pairs: auto

      # Scale targets so each track's genome-wide signal sums to this total,
      # as in AlphaGenome. rna_seq strand pairs share one factor.
      # CLI: --target-total 100M. Default: use the BigWig values as stored.
      # target_total: 100M

      # Alternative: global modality weights (same as task_weight per modality)
      # modality_weights: "atac:1.0,rna_seq:0.5,chip_tf:1.0"
      # or as dict:
      # modality_weights:
      #   atac: 1.0
      #   rna_seq: 0.5

      # =============================================================================
      # Training Configuration
      # =============================================================================

      epochs: 10                         # Number of training epochs
      batch_size: 1                      # Batch size per GPU
      gradient_accumulation_steps: 4     # Accumulate gradients over N batches

      # Learning rate and schedule
      lr: 0.0001                         # Learning rate
      weight_decay: 0.1                  # Weight decay for AdamW
      warmup_steps: 500                  # Linear warmup steps
      lr_schedule: cosine                # 'cosine' or 'constant'

      # Loss configuration
      positional_weight: 5.0             # Weight for positional (cross-entropy) loss
      count_weight: 1.0                  # Weight for count (Poisson) loss

      # Multinomial loss segmentation
      num_segments: 8                    # Number of segments for loss computation
      min_segment_size: 64               # Minimum segment size (optional)

      # Gradient clipping
      max_grad_norm: 1.0                 # Max gradient norm for clipping

      # Data loading
      num_workers: 4                     # DataLoader workers per GPU

      # Precision
      use_amp: true                      # Use automatic mixed precision (or no_amp: false)

      # Track means computation
      track_means_samples: null          # Samples for computing track means (null = all)

      # Compilation and profiling
      compile: false                     # Use torch.compile
      profile_batches: 0                 # Profile first N batches (0 = disabled)

      # Random seed
      seed: 42                           # Random seed (null for no seeding)

      # =============================================================================
      # Logging Configuration
      # =============================================================================

      wandb: true                        # Enable Weights & Biases logging
      wandb_project: alphagenome-finetune  # W&B project name
      wandb_entity: null                 # W&B entity (team/user)
      log_every: 50                      # Log every N batches

      # =============================================================================
      # Output Configuration
      # =============================================================================

      output_dir: finetuning_output      # Output directory
      run_name: my_experiment            # Run name (default: timestamp)
      save_every: 1                      # Save checkpoint every N epochs (0 = never)

      # =============================================================================
      # Resume Configuration
      # =============================================================================

      resume: null                       # Checkpoint path or 'auto' to find latest
      save_delta: false                  # Save delta checkpoints (adapter + head weights only)
      no_full_checkpoint: false          # With save_delta, skip full checkpoint files
      no_save_checkpoints: false         # Write no weights at all (keeps logs)

Combining a config file with CLI flags
---------------------------------------

The two can be mixed freely. **A flag you pass explicitly always wins**;
everything else falls back to the config file. This is decided per key, so you
can keep the stable parts of an experiment in YAML and vary one thing on the
command line:

.. code-block:: bash

   # config.yaml holds the data paths, model and mode; sweep the learning rate
   agt finetune --config config.yaml --lr 3e-5 --run-name lr3e5
   agt finetune --config config.yaml --lr 1e-4 --run-name lr1e4

Every key in the schema above can also be given as its CLI flag, with
underscores becoming dashes: ``lora_rank`` is ``--lora-rank``, ``output_dir`` is
``--output-dir``.

A few keys behave slightly differently:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Behaviour
   * - ``use_amp: true``
     - The inverse of ``--no-amp``. ``no_amp`` is also accepted.
   * - ``no_cache: true``
     - Shorthand that turns off both ``cache_genome`` and ``cache_signals``.
   * - ``modalities``
     - A nested mapping, not a scalar — see below.
   * - ``strand_pairs``
     - Set per modality as ``modalities.<name>.strand_pairs``, not at the top
       level. ``--strand-pairs`` overrides it for the modalities it names.

**Modalities.** Passing ``--modality``/``--bigwig`` pairs on the command line
*replaces* the BigWig list for those modalities, and adds any modality the
config did not mention. Modalities defined only in the config are left alone.
Per-modality settings such as ``resolutions`` and ``task_weight`` come from the
config either way, so a command line can override the file list while keeping
the config's per-modality tuning.

**Depth normalization.** Add ``--target-total 100M`` to a fine-tuning
command to scale targets in memory. Rank 0 scans the eligible BigWigs
before training and shares the factors with the other ranks. The training
and validation datasets apply the same factors before binning, and the head
uses track means in the corresponding units.

For stranded RNA-seq, add ``--strand-pairs rna_seq:auto`` when the BigWigs
are ordered plus/minus by sample. For example, if a pair starts with genome-wide
totals of 30M and 10M, both strands are multiplied by 2.5 and end up at
75M and 25M. CAGE and PRO-cap use separate factors for each strand.
ATAC and DNase are also scaled per track. The two ChIP modalities bypass
depth normalization so their fold-change units are preserved.

Each normalized track or pair needs a positive total. Normalized tracks
must contain non-negative signal. If a minus-strand BigWig encodes coverage
with negative values, flip its sign before using this flag.

The factor typically cancels when the loss divides targets by their track
means. If independently scaled tracks share an averaged mean through
``strand_pairs``, their relative weights in model space can change.
Predictions and metrics computed in signal space use the scaled units.
For an external evaluation, supply observed targets in the same units,
for example by scaling the original BigWigs with
``agt preprocess scale-bigwig``. Training logs and checkpoints record
the factors. Use the same inputs and normalization settings when resuming.

.. _what-a-run-produces:

What a training run produces
-----------------------------

Output goes to ``<output-dir>/<run-name>``, defaulting to
``finetuning_output/<timestamp>``. A **default run** writes:

.. code-block:: text

   finetuning_output/20260821_143022/
   ├── best_model.pth          # full checkpoint, rewritten whenever val loss improves
   ├── checkpoint_epoch1.pth   # full checkpoint, one per epoch (--save-every)
   ├── checkpoint_epoch2.pth
   ├── config.json             # the training hyperparameters for this run
   ├── training_log.csv        # per-step metrics
   ├── epoch_log.csv           # per-epoch scalar metrics
   └── epoch_track_metrics.csv # per-epoch track metrics

Two things about this are worth knowing before you start a long run:

- **Delta checkpoints are off by default.** A default run produces only full
  checkpoints (~1GB each). ``agt adapters export`` needs a *delta* checkpoint and
  rejects ``best_model.pth``, so if you intend to share the model, pass
  ``--save-delta``. See :doc:`checkpoints`.
- **No** ``transfer_config.json`` **is written.** The ``TransferConfig`` is embedded
  inside every checkpoint, so you do not normally need a separate copy.
  ``--export-transfer-config PATH`` writes one after training finishes.

Flags that change what is written:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Effect
   * - ``--save-delta``
     - Also write ``best_model.delta.pth`` and ``checkpoint_epoch{N}.delta.pth``.
       Not available with ``--mode full``.
   * - ``--no-full-checkpoint``
     - Write **only** deltas. Requires ``--save-delta``.
   * - ``--no-save-checkpoints``
     - Write **no weights at all** — logs and ``config.json`` only. For
       benchmarking or sweeps where the artifacts are not wanted. This also
       disables the preemption checkpoint.
   * - ``--save-every N``
     - Write a per-epoch checkpoint every N epochs instead of every epoch.
       ``--save-every 0`` writes no per-epoch checkpoints. The best-model
       checkpoint is still written. ``--resume auto`` then finds only a
       preemption checkpoint, so after a crash, pass the best-model checkpoint
       to ``--resume``.

.. note::

   ``--no-full-checkpoint`` and ``--no-save-checkpoints`` differ in scope:
   the first is an off switch for saving full checkpoints when delta checkpoints are saved,
   the second is an off switch for weight-saving entirely.

If the job is preempted (SIGUSR1), a ``checkpoint_preempt.pth`` is written so the
run can be resumed with ``--resume auto``.

Delta Checkpoints
-----------------

Use ``--save-delta`` to save lightweight delta checkpoints alongside full checkpoints.
Delta checkpoints contain only the trained weights (adapters + heads) and are much
smaller than full checkpoints:

.. code-block:: bash

   agt finetune --mode lora --save-delta \
       --genome hg38.fa \
       --modality atac --bigwig *.bw \
       --train-bed train.bed --val-bed val.bed \
       --pretrained-weights model.pth

This saves both:

- ``best_model.pth`` - Full checkpoint (~1GB)
- ``best_model.delta.pth`` - Delta checkpoint (~5-10MB for LoRA, ~1MB for linear-probe)

Add ``--no-full-checkpoint`` with ``--save-delta`` to write only delta checkpoints.
Delta checkpoints work with all modes except ``full`` (which trains all parameters).
To load a delta checkpoint, see :doc:`python_api`.

Supported Modalities
--------------------

.. list-table::
   :header-rows: 1
   :widths: 15 35 25 15

   * - Modality
     - Description
     - Default Resolutions
     - Squashing
   * - ``atac``
     - ATAC-seq chromatin accessibility
     - 1bp, 128bp
     - No
   * - ``dnase``
     - DNase-seq chromatin accessibility
     - 1bp, 128bp
     - No
   * - ``procap``
     - PRO-cap transcription
     - 1bp, 128bp
     - No
   * - ``cage``
     - CAGE transcription
     - 1bp, 128bp
     - No
   * - ``rna_seq``
     - RNA-seq gene expression (any library; see :ref:`rna-3p-libraries`)
     - 1bp, 128bp
     - Yes
   * - ``chip_tf``
     - ChIP-seq transcription factors
     - 128bp only
     - No
   * - ``chip_histone``
     - ChIP-seq histone modifications
     - 128bp only
     - No

Multi-Modality Training
-----------------------

Train on multiple assay types simultaneously using the ``modalities`` config section
or repeating ``--modality`` and ``--bigwig`` pairs on the CLI:

.. code-block:: bash

   agt finetune --mode lora \
       --genome hg38.fa \
       --pretrained-weights model.pth \
       --train-bed train.bed --val-bed val.bed \
       --modality atac --bigwig sample1_atac.bw sample2_atac.bw \
       --modality rna_seq --bigwig sample1_rna.bw \
       --modality-weights "atac:1.0,rna_seq:0.5"

Alternatively, use the matching YAML config:

.. code-block:: yaml

   modalities:
     atac:
       bigwig:
         - sample1_atac.bw
         - sample2_atac.bw
       task_weight: 1.0

     rna_seq:
       bigwig:
         - samplel1_rna.bw
       task_weight: 0.5

Multi-GPU Training
------------------

Use ``torchrun`` for distributed training. It needs a module target, so invoke
the CLI with ``-m``:

.. code-block:: bash

   torchrun --nproc_per_node=4 -m alphagenome_pytorch.cli finetune --mode lora ...

.. _gene-level-rna-seq:

Gene-Level RNA-seq
------------------

Two optional, independent features aggregate the ``rna_seq`` head over genes: a
**cross-track gene LFC loss** during training, and a **gene-expression
correlation metric** during validation. Both are off by default.

.. _rna-3p-libraries:

3'-biased libraries
^^^^^^^^^^^^^^^^^^^

A 10x-style pseudobulk track is still the ``rna_seq`` head. What changes is how
signal is reduced inside a gene body, and that is one setting:

* **mean** (the default) divides by gene length. Correct for full-length
  libraries, where coverage is spread along the transcript, so signal per bp is
  the expression estimate.
* **sum** keeps total signal. Correct for 3'-biased libraries, where reads pile
  up at the 3' end. Dividing by gene length would inject a spurious 1/length
  weight — and in the loss it down-weights long genes in *both* terms, not just
  the Poisson one.

Set it per head in a config, or run-wide on the command line:

.. code-block:: yaml

   modalities:
     rna_seq:
       bigwig: [mic_plus.bw, mic_minus.bw]
       strand: "+-"
       gene_reduce_func: sum     # 3'-biased library

.. code-block:: bash

   agt finetune --modality rna_seq --bigwig mic_plus.bw mic_minus.bw \
       --gene-reduce-func sum ...

The setting selects the gene LFC training reduction and the primary
``--gene-expr-eval`` metric. In ``sum`` mode the training loss uses gene totals
and divides the two loss terms by a fixed reference gene length of 20,000 bp.
Each gene then contributes as it would in ``mean`` mode if it were 20 kb long.
Start with the ``--gene-loss-weight`` that you would use in ``mean`` mode. The
reference length does not depend on the expression data. It was measured with
131 kb windows and a human annotation. For much larger windows or another
organism, the typical gene length can change, so adjust
``--gene-loss-weight``. Evaluation uses exon totals directly. An explicit CLI
reduction overrides the per-head config. ``auto`` uses the per-head config or
defaults to ``mean``.

Keeping the head named ``rna_seq`` is what lets the fine-tune be served and
scored: a fine-tuned head replaces the pretrained slot of the same name, while
serving and ``agt score`` address heads by output type, not by your name for
the library.

Gene LFC loss
^^^^^^^^^^^^^

A Decima-style cross-track log-fold-change term over gene bodies (exons +
introns), added to the per-position loss for the ``rna_seq`` head.

.. code-block:: bash

   agt finetune --mode lora \
       --genome hg38.fa --pretrained-weights model.pth \
       --train-bed train.bed --val-bed val.bed \
       --modality rna_seq --bigwig rna_plus.bw rna_minus.bw \
       --track-strands '+-' \
       --gtf gencode.v46.annotation.gtf \
       --gene-loss-weight 0.1

======================================= =========== ==================================================
Flag                                    Default     Meaning
======================================= =========== ==================================================
``--gene-loss-weight W``                ``0.0``     Outer weight on the gene LFC term (paper: ``0.1``).
                                                    ``0.0`` disables the term entirely.
``--gene-cross-track-weight W``         ``5.0``     Inner multinomial weight (paper default). Only
                                                    used when ``--gene-loss-weight > 0``.
``--gene-reduce-func MODE``             ``auto``    ``auto`` | ``mean`` | ``sum``. Within-gene
                                                    reduction for the loss *and* the metric;
                                                    ``auto`` takes the per-head config value, else
                                                    ``mean`` (see :ref:`rna-3p-libraries`).
``--gtf PATH``                          *(none)*    Annotation, parquet or GTF;
                                                    ``protein_coding`` **gene** rows build the
                                                    per-window gene-body masks. Prefer parquet —
                                                    it loads in seconds where a GENCODE GTF takes
                                                    minutes, on every run's startup path.
``--track-strands STR``                 *(none)*    One char per ``rna_seq`` BigWig: ``+-+-`` or
                                                    ``+,-,+,-``. YAML: ``modalities.rna_seq.strand``.
======================================= =========== ==================================================

Enabling it requires all three of ``--gtf``, ``rna_seq`` in ``--modality``, and
``--track-strands``. Leaving ``--gene-loss-weight`` at ``0.0`` keeps loss values
bit-identical to a run without the feature.

Gene-expression validation metric
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--gene-expr-eval`` reports how well predicted gene expression correlates with
the observed tracks each validation epoch. The quantity is the AlphaGenome
definition — log-transformed coverage over a gene's **annotated exons**,
strand-matched, keeping genes with at least 50% of their exons inside the
window. Exons, not the gene bodies the LFC loss aggregates; the within-gene
reduction, however, is the head's own ``gene_reduce_func``, so the metric
scores what the loss optimizes.

.. code-block:: bash

   agt finetune --mode lora \
       --genome hg38.fa --pretrained-weights model.pth \
       --train-bed train.bed --val-bed val.bed \
       --modality rna_seq --bigwig rna_plus.bw rna_minus.bw \
       --track-strands '+-' --strand-pairs 'rna_seq:0,1' \
       --gene-expr-eval --gene-expr-annotation gencode.v46.parquet

Emitted metric keys. The three ``*_pearson_*`` keys appear in the per-epoch
console summary and are logged verbatim; ``n_genes`` is logged as
``val_loss_rna_seq_gene_log_expr_n_genes``:

======================================================= =====================================================
Key                                                     Meaning
======================================================= =====================================================
``rna_seq_gene_log_expr_pearson_across_genes``          Raw Pearson over genes, one per track, averaged
``rna_seq_gene_log_expr_pearson_across_genes_norm``     Same after quantile normalization + gene-mean
                                                        centering (specificity view)
``rna_seq_gene_log_expr_pearson_across_tracks_norm``    Normalized data correlated per gene across tracks
``rna_seq_gene_log_expr_pearson_one_vs_rest_lfc``       One-vs-rest log2 fold change, per track, averaged
``rna_seq_gene_log_expr_pearson_{mean,sum}_*``           The three ``across_*`` keys above under **both**
                                                        within-gene reductions, always reported, so runs
                                                        on different library types stay comparable. The
                                                        unsuffixed keys repeat the head's own reduction.
``rna_seq_gene_log_expr_n_genes``                       Genes contributing, deduplicated across windows
======================================================= =====================================================

The one-vs-rest fold change is emitted once, without a reduction suffix: every
track of a gene is divided by the same gene length, so it cancels in the ratio
and both reductions give the identical number.

One-vs-rest fold change
"""""""""""""""""""""""

For each track, the ratio of that track to the mean of the remaining tracks —
formed *within* the predictions and *within* the observations, then correlated
across genes. It is computed automatically whenever the metric runs and there
are at least two tracks, and is also emitted per track as
``..._one_vs_rest_lfc/<track name>``.

Watch it alongside ``across_genes``. Absolute expression is largely inherited
from the pretrained trunk, which already knows which genes are highly expressed;
the differential between samples is what an adapter has to add. A run where
``across_genes`` looks healthy while this stays near zero has learned nothing
cell-type-specific.

Stranded gene-expression metrics require explicit sample identities. Set
``strand_pair_id`` to the same value on a sample’s two rows in the
``--track-metadata`` parquet, or pass explicit zero-based plus/minus indices
with ``--strand-pairs 'rna_seq:0,3;1,2'``. Each stranded track must have exactly
one partner. Full and delta training checkpoints save explicit pair indices
for each head, even when ``--gene-expr-eval`` is disabled. The evaluator uses
those pairs if present, then falls back to the recorded ``strand_pair_id``
values. Pass ``--strand-pairs '0,3;1,2'`` to override either source.
Its ``*_one_vs_rest_lfc_by_sample`` output includes the sample ID, zero-based
source track indices, and correlation.

The evaluator uses the checkpoint's recorded ``gene_reduce_func`` for the
unsuffixed metrics and reports the selected reduction in its JSON output.
Older checkpoints default to ``mean``. Pass ``--gene-reduce-func mean`` or
``--gene-reduce-func sum`` to override the default.

Per-track metric series
^^^^^^^^^^^^^^^^^^^^^^^

``epoch_track_metrics.csv`` records a row for every track and metric at each
epoch, including wide heads. Count and profile correlations use the original
output tracks. Gene fold-change correlations use one sense column per sample.
``epoch_log.csv`` contains the scalar summaries. W&B shows named series for
heads with up to 32 tracks and histograms for wider heads.

This is what distinguishes "three good ``+`` tracks and three dead ``-`` tracks"
from a uniform ceiling across every track — the first means the targets or the
strand wiring are wrong, the second means capacity, learning rate or context.
Both show up at the first validation epoch, so a broken run can be stopped
there rather than after every epoch has been spent.

Requirements that are checked before training starts:

- the annotation has **exon rows**. ``--gene-expr-annotation`` takes it, falling
  back to ``--gtf``. A stock GENCODE annotation has exon rows, so the fallback
  normally works; pass ``--gene-expr-annotation`` only when your ``--gtf`` is a
  gene-only file, which is all the LFC loss needs. An annotation without exon
  rows is rejected.

  Both flags take parquet or GTF, and both are much faster on parquet.
  Convert once with ``scripts/convert_gtf_to_parquet.py`` and point ``--gtf`` at
  the result: it preserves all the features, so a single file will work for the gene-body
  masks and the exon metric (no extra ``--gene-expr-annotation`` necessary).
- ``rna_seq`` in ``--modality`` / config
- per-track strands via ``--track-strands`` or ``modalities.rna_seq.strand``

Genes are deduplicated by id across validation windows, so overlapping windows
do not double-count. Under DDP the per-window values are gathered across ranks
before the correlations are computed.

YAML equivalents:

.. code-block:: yaml

   # One parquet with exon rows drives both features; gene_expr_annotation is
   # only needed when this file is gene-only.
   gtf: /data/annotation/gencode.v46.parquet
   gene_loss_weight: 0.1
   gene_cross_track_weight: 5.0

   gene_expr_eval: true

   modalities:
     rna_seq:
       bigwig:
         - rna_plus.bw
         - rna_minus.bw
       strand: "+-"

The same exon aggregation is available outside training — for a per-gene count
table from a finetuned checkpoint, see :ref:`per-gene-counts`; for the
underlying functions, :doc:`/api/aggregation`.

Example Configurations
----------------------

**Minimal single-modality config:**

.. code-block:: yaml

   genome: hg38.fa
   train_bed: train.bed
   val_bed: val.bed
   pretrained_weights: model.pth

   modalities:
     atac:
       bigwig:
         - sample1.bw
         - sample2.bw

**Full-featured multi-modality config:**

.. code-block:: yaml

   genome: /data/genomes/hg38.fa
   train_bed: /data/beds/train_peaks.bed
   val_bed: /data/beds/val_peaks.bed
   pretrained_weights: /models/alphagenome_v1.pth

   output_dir: /output/multitask_experiment
   run_name: atac_rna_chip_v1

   mode: lora
   lora_rank: 8
   lora_alpha: 16
   gradient_checkpointing: true

   epochs: 20
   batch_size: 1
   gradient_accumulation_steps: 8
   lr: 1e-4
   warmup_steps: 1000

   positional_weight: 5.0
   count_weight: 1.0

   wandb: true
   wandb_project: alphagenome-multitask

   modalities:
     atac:
       bigwig:
         - /data/bigwigs/atac_s1.bw
         - /data/bigwigs/atac_s2.bw
         - /data/bigwigs/atac_s3.bw
       resolutions: "1,128"
       task_weight: 1.0

     rna_seq:
       bigwig:
         - /data/bigwigs/rna_s1.bw
         - /data/bigwigs/rna_s2.bw
       resolutions: "128"
       task_weight: 0.5

Generating Predictions (BigWig)
-------------------------------

After training, generate chromosome-wide predictions with ``agt predict``.

A **delta checkpoint** holds only the difference from the base model, so pass
both — the base weights as ``--model`` and the fine-tune as ``--checkpoint``:

.. code-block:: bash

   agt predict \
       --model pretrained.pth \
       --checkpoint best_model.delta.pth \
       --fasta hg38.fa \
       --output predictions/ \
       --head my_atac \
       --chromosomes chr21

A **full checkpoint** contains the whole model, so ``--model`` is optional:

.. code-block:: bash

   agt predict \
       --checkpoint best_model.pth \
       --fasta hg38.fa \
       --output predictions/ \
       --head my_atac \
       --chromosomes chr21

``--checkpoint`` also accepts an adapter bundle directory or an ``hf://`` URI.
See :doc:`checkpoints` for every artifact kind and how to load it.

.. code-block:: bash

   # Legacy full checkpoint whose TransferConfig was not embedded
   agt predict \
       --model pretrained.pth \
       --checkpoint best_model.pth \
       --transfer-config transfer_config.json \
       --fasta hg38.fa \
       --output predictions/ \
       --head my_atac

The transfer config is embedded in checkpoints but you can
also export it from a training run as a separate file:

.. code-block:: bash

   agt finetune ... --export-transfer-config transfer_config.json

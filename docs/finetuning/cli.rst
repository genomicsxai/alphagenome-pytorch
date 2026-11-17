Command-Line Interface
======================

``agt finetune`` is the entry point for fine-tuning using the CLI,
and it can be configured with CLI flags, a YAML config file, or both.

.. note::

   ``agt finetune`` is analogous to running ``python scripts/finetune.py``
   but it works system-wide and ships with the installed package.


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

      # Model precision
      dtype: bfloat16                    # 'bfloat16' or 'float32'

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
      save_every: 1                      # Save checkpoint every N epochs

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
   └── epoch_log.csv           # per-epoch metrics

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
     - RNA-seq gene expression
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

Two optional, independent features aggregate the ``rna_seq`` head over genes:
a **cross-track gene LFC loss** during training, and a **gene-expression
correlation metric** during validation. Both are off by default.

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
``--gtf PATH``                          *(none)*    Annotation, parquet or GTF/GFF;
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
definition — log-transformed mean coverage over a gene's **annotated exons**,
strand-matched, keeping genes with at least 50% of their exons inside the
window — not the gene-body aggregation the LFC loss uses.

.. code-block:: bash

   agt finetune --mode lora \
       --genome hg38.fa --pretrained-weights model.pth \
       --train-bed train.bed --val-bed val.bed \
       --modality rna_seq --bigwig rna_plus.bw rna_minus.bw \
       --track-strands '+-' \
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
``rna_seq_gene_log_expr_n_genes``                       Genes contributing, deduplicated across windows
======================================================= =====================================================

Requirements that are checked before training starts:

- the annotation has **exon rows**. ``--gene-expr-annotation`` takes it, falling
  back to ``--gtf``. A stock GENCODE annotation has exon rows, so the fallback
  normally works; pass ``--gene-expr-annotation`` only when your ``--gtf`` is a
  gene-only file, which is all the LFC loss needs. An annotation without exon
  rows is rejected.

  Both flags take parquet or GTF/GFF, and both are much faster on parquet.
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

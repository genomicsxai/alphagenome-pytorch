Command-Line Interface
======================

The unified training script ``scripts/finetune.py`` supports all training modes
and can be configured via CLI arguments or YAML config files.

Multi-GPU Training
------------------

Use ``torchrun`` for distributed training:

.. code-block:: bash

   torchrun --nproc_per_node=4 scripts/finetune.py --mode lora ...

YAML Configuration
------------------

For reproducible experiments, use ``--config config.yaml``. CLI arguments
override YAML values when both are provided.

.. code-block:: bash

   pip install pyyaml
   python scripts/finetune.py --config config.yaml

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
      # Baskerville-style Locon parity uses 'lora+locon'
      mode: lora

      # LoRA configuration (used when mode includes LoRA)
      lora_rank: 8                       # LoRA rank (0 disables LoRA, trains heads only)
      lora_alpha: 16                     # LoRA alpha scaling factor
      lora_targets: "q_proj,v_proj"      # Comma-separated list of target modules

      # Locon configuration (used when mode includes Locon)
      locon_rank: 4
      locon_alpha: 1
      locon_targets: "down_blocks.4,down_blocks.5"  # Required; Locon4 on encoder blocks

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
            - /path/to/sample1_rna.bw
          resolutions: "128"             # RNA-seq at 128bp only
          task_weight: 0.5               # Lower weight for RNA-seq

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

Delta Checkpoints
-----------------

Use ``--save-delta`` to save lightweight delta checkpoints alongside full checkpoints.
Delta checkpoints contain only the trained weights (adapters + heads) and are much
smaller than full checkpoints:

.. code-block:: bash

   python scripts/finetune.py --mode lora --save-delta \
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

   python scripts/finetune.py --mode lora \
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

.. _gene-level-rna-seq:

Gene-Level RNA-seq
------------------

Two optional, independent features aggregate the ``rna_seq`` head over genes:
a **cross-track gene LFC loss** during training, and a **gene-expression
correlation metric** during validation. Both are off by default.

The commands below use ``agt finetune``; it is the installed entry point for the
same code as ``python scripts/finetune.py`` used elsewhere on this page.

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
console summary and are logged verbatim to Weights & Biases; ``n_genes`` takes
the generic path and is logged as
``val_loss_rna_seq_gene_log_expr_n_genes`` (a naming artifact — it is a count,
not a loss):

======================================================= =====================================================
Key                                                     Meaning
======================================================= =====================================================
``rna_seq_gene_log_expr_pearson_across_genes``          Raw Pearson over genes, one per track, averaged
``rna_seq_gene_log_expr_pearson_across_genes_norm``     Same after quantile normalization + gene-mean
                                                        centering (specificity view)
``rna_seq_gene_log_expr_pearson_across_tracks_norm``    Normalized data correlated per gene across tracks
``rna_seq_gene_log_expr_n_genes``                       Genes contributing, deduplicated across windows
======================================================= =====================================================

Requirements, all checked before training starts:

- an annotation **with exon rows**. ``--gene-expr-annotation`` takes it, falling
  back to ``--gtf``. A stock GENCODE annotation has exon rows, so the fallback
  normally works; pass ``--gene-expr-annotation`` only when your ``--gtf`` is a
  gene-only file, which is all the LFC loss needs. An annotation without exon
  rows is rejected.

  Both flags take parquet or GTF/GFF, and both are much faster on parquet.
  Convert once with ``scripts/convert_gtf_to_parquet.py`` and point ``--gtf`` at
  the result: it preserves every feature, so a single file drives the gene-body
  masks and the exon metric, and ``--gene-expr-annotation`` becomes unnecessary.
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

After training, generate chromosome-wide predictions using
``scripts/predict_full_chromosome.py``. Pass your base pretrained weights
as ``--model`` and the finetuned checkpoint as ``--checkpoint``:

.. code-block:: bash

   # Delta checkpoint
   python scripts/predict_full_chromosome.py \
       --model pretrained.pth \
       --checkpoint best_model.delta.pth \
       --fasta hg38.fa \
       --output predictions/ \
       --head my_atac \
       --chromosomes chr21

   # Full checkpoint (with embedded TransferConfig)
   python scripts/predict_full_chromosome.py \
       --model pretrained.pth \
       --checkpoint best_model.pth \
       --fasta hg38.fa \
       --output predictions/ \
       --head my_atac \
       --chromosomes chr21

   # Full checkpoint (with external TransferConfig)
   python scripts/predict_full_chromosome.py \
       --model pretrained.pth \
       --checkpoint best_model.pth \
       --transfer-config transfer_config.json \
       --fasta hg38.fa \
       --output predictions/ \
       --head my_atac

The transfer config is embedded in checkpoints but you can 
also export it from a training run as a separate file:

.. code-block:: bash

   python scripts/finetune.py ... --export-transfer-config transfer_config.json

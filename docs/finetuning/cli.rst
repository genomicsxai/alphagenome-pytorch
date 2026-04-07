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

      # Training mode: 'linear-probe', 'lora', or 'full'
      # Python API also supports 'locon', 'ia3', and combined modes (e.g. ['lora', 'locon'])
      mode: lora

      # LoRA configuration (only used when mode='lora')
      lora_rank: 8                       # LoRA rank (0 disables LoRA, trains heads only)
      lora_alpha: 16                     # LoRA alpha scaling factor
      lora_targets: "q_proj,v_proj"      # Comma-separated list of target modules

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

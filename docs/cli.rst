Command-Line Interface (``agt``)
================================

AlphaGenome PyTorch ships a CLI called ``agt`` — short for
**A**\ lphaGenome **T**\ orch (and three of the four nucleotides).

After installing the package the command is available globally:

.. code-block:: bash

   pip install alphagenome-pytorch          # minimal — info/convert only
   pip install alphagenome-pytorch[inference]  # + predict
   pip install alphagenome-pytorch[finetuning] # + finetune
   pip install alphagenome-pytorch[scoring]    # + score

Global options
--------------

.. code-block:: text

   agt [--json] <command> [options]

``--json``
   Machine-readable JSON output on stdout.  Suppresses progress bars and
   human formatting.  Long-running commands (``predict``, ``finetune``)
   emit JSONL (one JSON object per line).

   Errors produce a JSON object on stderr with a nonzero exit code:

   .. code-block:: json

      {"error": "FileNotFoundError", "message": "No such file: model.pth"}


``agt info``
------------

Inspect the model architecture, available heads, track metadata, and the
contents of a weights file.

Static information (no weights file needed)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Overview — heads, track counts per organism, resolutions
   agt info

   # List all heads with track counts
   agt info --heads

Example output:

.. code-block:: text

   Head              Tracks (human)  Tracks (mouse)  Dimension  Resolutions
   atac                         167             155        256  1bp, 128bp
   dnase                        305             280        384  1bp, 128bp
   procap                        12               8        128  1bp, 128bp
   cage                         546             490        640  1bp, 128bp
   rna_seq                      667             600        768  1bp, 128bp
   chip_tf                     1617            1500       1664  128bp
   chip_histone                1116            1000       1152  128bp
   contact_maps                  28              28         28  64x64
   splice_sites                   5               5          5  1bp
   splice_junctions             734             734        734  pairwise
   splice_site_usage            734             734        734  1bp

*Tracks* = real (non-padding) tracks per organism.
*Dimension* = tensor channel size (includes padding).

.. code-block:: bash

   # List individual tracks for a head
   agt info --tracks atac
   agt info --tracks atac --organism mouse

   # Search tracks by name or metadata
   agt info --tracks atac --search K562
   agt info --tracks atac --filter "biosample_name=liver"

Example output:

.. code-block:: text

   Head: atac | 167 tracks / 256 dimension (89 padding) | human

     #   Track Name                     Biosample       Ontology
     0   ENCSR637XSC ATAC-seq           K562            EFO:0002067
     1   ENCSR868FGM ATAC-seq           HepG2           EFO:0001187
     ...
   166   UBERON:0015143 ATAC-seq        thymus           UBERON:0015143
         --- 89 padding tracks ---

Weights file inspection
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Inspect a weights file — adds: file size, param count, dtype, format
   agt info model.pth

   # Inspect track_means for a specific head
   agt info model.pth --track-means atac
   agt info model.pth --track-means atac --organism human --top 10

   # Validate a checkpoint — checks all keys present, shapes match
   agt info model.pth --validate

   # Compare two checkpoints
   agt info model.pth --diff other.pth

   # Inspect a delta/finetuned checkpoint
   agt info delta.safetensors

JSON output
^^^^^^^^^^^

.. code-block:: bash

   agt --json info --heads

.. code-block:: json

   {
     "heads": [
       {
         "name": "atac",
         "dimension": 256,
         "tracks": {"human": 167, "mouse": 155},
         "padding": {"human": 89, "mouse": 101},
         "resolutions": ["1bp", "128bp"]
       }
     ]
   }

.. code-block:: bash

   agt --json info model.pth

.. code-block:: json

   {
     "file": "model.pth",
     "format": "pth",
     "file_size_mb": 1247.3,
     "total_parameters": 298542080,
     "dtype": "float32",
     "has_track_means": true,
     "heads": ["atac", "dnase", "procap", "cage", "rna_seq", "chip_tf", "chip_histone"]
   }


``agt predict``
---------------

Run the model and write predictions to disk. Four input modes:

=================== =============================================== ========================
Input mode          What it does                                    Output
=================== =============================================== ========================
``--chromosomes``   Full-chromosome tiling                          BigWig per track
``--locus``         One genomic interval                            BigWig per track
``--bed``           Many genomic regions from a BED file            BigWig per track (merged)
``--sequences``     Raw FASTA sequences (no genomic coordinates)    NPZ per sequence
=================== =============================================== ========================

``--locus``, ``--bed``, and ``--sequences`` are mutually exclusive. When
``--bed`` is given, ``--chromosomes`` can additionally be passed as a
chromosome filter over the BED rows (see below).

Requires: ``pip install alphagenome-pytorch[inference]``

How size mismatches are handled
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The model expects a fixed input window of ``W`` bp, where ``W`` is set by
``--window-size`` (default: 131 072). When an input region or sequence does
not match ``W``, the CLI dispatches by mode and the ``--tile`` flag:

+------------------------+------------------------------+-------------------------+------------------------------+
| Mode                   | Input < ``W``                | Input == ``W``          | Input > ``W``                |
+========================+==============================+=========================+==============================+
| ``--locus`` / ``--bed``| padded with real reference   | single window           | **cut** to center            |
| (default)              | flanks (with warning)        |                         | (with warning)               |
+------------------------+------------------------------+-------------------------+------------------------------+
| ``--locus`` / ``--bed``| padded with real reference   | single window           | stitched tiles               |
| ``--tile``             | flanks (with warning)        |                         |                              |
+------------------------+------------------------------+-------------------------+------------------------------+
| ``--sequences``        | **error** (cannot fake       | single window           | **error** (pass ``--tile``)  |
| (default)              | reference context)           |                         |                              |
+------------------------+------------------------------+-------------------------+------------------------------+
| ``--sequences``        | **error**                    | single window           | stitched tiles               |
| ``--tile``             |                              |                         |                              |
+------------------------+------------------------------+-------------------------+------------------------------+
| ``--chromosomes``      | n/a                          | n/a                     | stitched tiles (always)      |
+------------------------+------------------------------+-------------------------+------------------------------+

Input validation
""""""""""""""""

Chromosome coordinates must be non-negative and must fit inside the
chromosome. The CLI rejects invalid input up front rather than clamping
silently:

.. code-block:: text

   Error: Invalid locus 'chr1:-100-500': start (-100) must be ≥ 0
   Error: chr1:248000000-250000000: end (250000000) exceeds chromosome length (248956422)

When a short region is padded, the fitted ``W``-bp window is also required
to be in-bounds. If the region sits near a chromosome edge the window is
shifted inward (never clamped to negative coordinates), and a warning
describes the shift.

Per-region logging
""""""""""""""""""

Each processed region prints a one-line status to stdout (suppressed under
``--quiet`` / ``--json``), plus any warning lines to stderr:

.. code-block:: text

   chr2:5000-7000        (2000bp)    → padded
     WARNING: chr2:5000-7000 (2000bp) padded with reference flanks; window shifted to [0, 131072) because region sits near chromosome start.
   chr3:10000000-10002000 (2000bp)   → padded
     WARNING: chr3:10000000-10002000 (2000bp) padded with reference flanks to a 131072bp window [9935464, 10066536); output covers only the region.
   chr4:1000000-2000000  (1000000bp) → tiled (12 tiles)
   chr4:100-131172       (131072bp)  → single
   chr5:50000-1050000    (1000000bp) → cut
     WARNING: chr5:50000-1050000 (1000000bp) center-cut to chr5:484464-615536 (131072bp); pass --tile to predict the full region.

Full chromosomes
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Predict ATAC for chr1 and chr2
   agt predict \
       --model model.pth --fasta hg38.fa --output predictions/ \
       --head atac --chromosomes chr1,chr2

   # Whole genome, 1bp resolution (slower), torch.compile for speed
   agt predict \
       --model model.pth --fasta hg38.fa --output predictions/ \
       --head atac --chromosomes chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10 \
       --resolution 1 --compile

   # Reduce edge artifacts with overlapping tiles
   agt predict \
       --model model.pth --fasta hg38.fa --output predictions/ \
       --head atac --chromosomes chr1 --crop-bp 32768

Locus (single interval)
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Exactly one window — single forward pass
   agt predict \
       --model model.pth --fasta hg38.fa --output out/ \
       --head atac --locus chr1:10000000-10131072

   # Short locus — padded with real reference flanks
   agt predict \
       --model model.pth --fasta hg38.fa --output out/ \
       --head atac --locus chr1:10000000-10005000

   # Long locus (default) — center-cut to 131 072 bp, warning printed
   agt predict \
       --model model.pth --fasta hg38.fa --output out/ \
       --head atac --locus chr1:10000000-11000000

   # Long locus with --tile — full region predicted via stitched tiles
   agt predict \
       --model model.pth --fasta hg38.fa --output out/ \
       --head atac --locus chr1:10000000-11000000 \
       --tile --crop-bp 16384

Output file name:
``{output}/{head}_{chrom}_{start}_{end}.bw`` (or per-track when multiple tracks).

BED file (many regions)
^^^^^^^^^^^^^^^^^^^^^^^

BED columns: ``chrom``, ``start``, ``end``, optional ``name``. Lines
starting with ``#``, ``track``, or ``browser`` are skipped. Each region is
processed independently with the same size-handling rules as ``--locus``;
all region predictions are merged into a single BigWig per track (gaps
between regions are left as no-data).

.. code-block:: bash

   # Default — short regions padded, exact = single, long = cut (warns)
   agt predict \
       --model model.pth --fasta hg38.fa --output out/ \
       --head atac --bed regions.bed

   # --tile — long regions are stitched instead of cut
   agt predict \
       --model model.pth --fasta hg38.fa --output out/ \
       --head atac --bed regions.bed --tile --crop-bp 16384

   # Filter: predict only the rows on chr1 / chr2
   agt predict \
       --model model.pth --fasta hg38.fa --output out/ \
       --head atac --bed regions.bed --chromosomes chr1,chr2

When ``--chromosomes`` is passed alongside ``--bed`` it acts as a whitelist
filter on the BED rows — useful for re-running a per-chromosome subset
without having to edit the BED. If the filter removes every row the CLI
errors out.

Output file name: ``{output}/{head}.bw`` (or per-track).

Raw FASTA sequences
^^^^^^^^^^^^^^^^^^^

Predict on arbitrary DNA that isn't tied to a reference genome. No
``--fasta`` (reference) is needed — the sequences themselves are the input.
Because there is no genome to fetch flanks from, short sequences are
rejected outright rather than N-padded; pre-pad them yourself if you need
that.

.. code-block:: bash

   # Sequences exactly the window size — one forward pass each
   agt predict \
       --model model.pth --output out/ \
       --head atac --sequences window_sized.fa

   # Longer sequences — tiling must be explicit
   agt predict \
       --model model.pth --output out/ \
       --head atac --sequences long_seqs.fa --tile --crop-bp 16384

Output: one ``{head}_{seq_name}.npz`` file per sequence (with metadata)
plus a ``manifest.json`` listing all outputs when in ``--json`` mode.

Errors you'll see:

.. code-block:: text

   Error: Sequence 'seq2' (5000bp) is shorter than the model window
   (131072bp); not supported for --sequences.

   Error: Sequence 'seq1' (500000bp) is longer than the model window
   (131072bp); pass --tile to enable tiling.

Common options
^^^^^^^^^^^^^^

======================== =================================================================
Flag                     Meaning
======================== =================================================================
``--head NAME``          Prediction head (``atac``, ``dnase``, ``cage``, …)
``--tracks 0,1,2``       Comma-separated track indices (default: all tracks)
``--track-names``        Comma-separated output track names
``--resolution {1,128}`` Output resolution in bp (128 is faster)
``--crop-bp N``          Per-tile edge crop; use with ``--tile`` to reduce edge artifacts
``--batch-size N``       Inference batch size (default 4)
``--window-size N``      Override model input window (default 131 072)
``--organism {0,1}``     0 = human, 1 = mouse
``--device STR``         PyTorch device (``cuda``, ``cpu``, ``mps``)
``--dtype-policy``       ``full_float32`` (default) or ``mixed_precision``
``--compile``            Wrap model with ``torch.compile``
``--checkpoint PATH``    Finetuned checkpoint (LoRA, full, etc.)
``--transfer-config``    TransferConfig JSON for adapter models
``--no-merge-adapters``  Keep LoRA/adapter modules separate from base weights
``--quiet``              Suppress progress bars and per-region status lines
======================== =================================================================

JSON output
^^^^^^^^^^^

For ``--locus`` / ``--bed``:

.. code-block:: json

   {
     "output_files": [
       {
         "path": "out/atac_chr1_10000000_10131072.bw",
         "head": "atac",
         "chromosome": "chr1",
         "start": 10000000,
         "end": 10131072,
         "length_bp": 131072,
         "handling": "single",
         "tile_count": 1,
         "resolution_bp": 128
       }
     ],
     "warnings": []
   }

For ``--bed`` an additional ``regions`` array lists per-region metadata
(``handling``, ``tile_count``, ``warnings``).

For ``--sequences``:

.. code-block:: json

   {
     "output_files": [
       {
         "path": "out/atac_seq1.npz",
         "head": "atac",
         "sequence": "seq1",
         "length_bp": 500000,
         "handling": "tiled",
         "tile_count": 4,
         "resolution_bp": 128
       }
     ],
     "warnings": []
   }


``agt finetune``
----------------

Training and finetuning — supports linear probing, LoRA, full
finetuning, and encoder-only modes.

Requires: ``pip install alphagenome-pytorch[finetuning]``

.. code-block:: bash

   # Linear probing (frozen backbone)
   agt finetune --mode linear-probe \
       --genome hg38.fa \
       --modality atac --bigwig *.bw \
       --train-bed train.bed --val-bed val.bed \
       --pretrained-weights model.pth \
       --resolutions 1

   # LoRA finetuning
   agt finetune --mode lora \
       --lora-rank 8 --lora-alpha 16 \
       --genome hg38.fa \
       --modality atac --bigwig *.bw \
       --train-bed train.bed --val-bed val.bed \
       --pretrained-weights model.pth \
       --resolutions 1

   # Encoder-only (CNN encoder, no transformer)
   agt finetune --mode encoder-only \
       --genome hg38.fa \
       --modality atac --bigwig *.bw \
       --train-bed train.bed --val-bed val.bed \
       --pretrained-weights model.pth \
       --sequence-length 500 --resolutions 128

   # Multi-modality
   agt finetune --mode lora \
       --genome hg38.fa \
       --modality atac --bigwig atac1.bw atac2.bw \
       --modality rna_seq --bigwig rna1.bw rna2.bw \
       --modality-weights atac:1.0,rna_seq:0.5 \
       --train-bed train.bed --val-bed val.bed \
       --pretrained-weights model.pth

JSON output (JSONL — one line per event):

.. code-block:: text

   {"event": "start", "mode": "lora", "lora_rank": 8, "total_params": 2457600}
   {"event": "step", "epoch": 1, "step": 100, "loss": 0.4231, "lr": 0.0001}
   {"event": "step", "epoch": 1, "step": 200, "loss": 0.3892, "lr": 0.0001}
   {"event": "validation", "epoch": 1, "val_loss": 0.3654, "pearson_r": 0.82}
   {"event": "checkpoint", "path": "checkpoints/epoch_1.pth"}
   {"event": "end", "best_val_loss": 0.3201, "best_epoch": 3}


``agt score``
-------------

Variant effect prediction — score the impact of genetic variants on
genomic tracks.

Requires: ``pip install alphagenome-pytorch[scoring]``

.. code-block:: bash

   # Score a single variant (format: chr:pos:ref>alt)
   agt score \
       --model model.pth \
       --fasta hg38.fa \
       --variant "chr22:36201698:A>C"

   # Score variants from a VCF
   agt score \
       --model model.pth \
       --fasta hg38.fa \
       --vcf variants.vcf \
       --head atac \
       --output scores.tsv

   # Score with specific aggregation method
   agt score \
       --model model.pth \
       --fasta hg38.fa \
       --vcf variants.vcf \
       --scorer logfc_max \
       --output scores.tsv

JSON output:

.. code-block:: json

   {
     "variants": [
       {
         "chrom": "chr22",
         "pos": 36201698,
         "ref": "A",
         "alt": "C",
         "scores": {
           "atac": {"logfc_max": 0.42, "logfc_mean": 0.11},
           "dnase": {"logfc_max": 0.38, "logfc_mean": 0.09}
         }
       }
     ]
   }


``agt convert``
---------------

Convert JAX AlphaGenome checkpoint to PyTorch format.

Requires: ``pip install alphagenome-pytorch[jax]``

.. code-block:: bash

   # Basic conversion
   agt convert --input /path/to/jax/checkpoint --output model.pth

   # Convert to safetensors format
   agt convert --input /path/to/jax/checkpoint --output model.safetensors

JSON output:

.. code-block:: json

   {
     "output": "model.pth",
     "format": "pth",
     "params_mapped": 1847,
     "params_total": 1847,
     "heads": ["atac", "dnase", "procap", "cage", "rna_seq", "chip_tf", "chip_histone"],
     "track_means_included": true
   }


``agt preprocess``
------------------

Data preprocessing utilities.  Each operation is a subcommand.

``bigwig-to-mmap``
^^^^^^^^^^^^^^^^^^

Convert BigWig files to memory-mapped format for fast training.

.. code-block:: bash

   agt preprocess bigwig-to-mmap \
       --input "*.bw" \
       --output training_data/ \
       --genome hg38.fa \
       --resolution 128

JSON output:

.. code-block:: json

   {
     "output_files": [
       {"path": "training_data/sample1.mmap", "tracks": 1, "size_mb": 234.5}
     ],
     "records_processed": 12345
   }

``scale-bigwig``
^^^^^^^^^^^^^^^^

Normalize BigWig signal to a target total (e.g. 100M reads).  Useful for
making tracks comparable before training or visualization.

The ``--target`` flag accepts human-readable suffixes: ``100M``, ``50M``,
``100k``, etc.

.. code-block:: bash

   # Scale a single file to 100M total signal
   agt preprocess scale-bigwig \
       --input sample.bw \
       --output sample_scaled.bw \
       --target 100M

   # Scale multiple files
   agt preprocess scale-bigwig \
       --input "*.bw" \
       --output scaled/ \
       --target 100M

   # Just compute the scale factor without writing output
   agt preprocess scale-bigwig \
       --input sample.bw \
       --target 100M \
       --dry-run

JSON output:

.. code-block:: json

   {
     "files": [
       {
         "input": "sample.bw",
         "output": "scaled/sample.bw",
         "original_total": 287453120.0,
         "target_total": 100000000.0,
         "scale_factor": 0.3479
       }
     ]
   }

``--dry-run`` returns the same JSON but skips writing output files.


``agt serve``
-------------

Serve the model via REST or gRPC.

.. note::

   Not yet implemented. This command is reserved for a future release.

.. code-block:: bash

   agt serve --model model.pth --port 8080

.. code-block:: text

   Error: 'agt serve' is not yet implemented.
   Follow https://github.com/user/alphagenome-pytorch for updates.


Dependency Gating
-----------------

Each subcommand checks for its required optional dependencies at runtime
and prints an actionable error message if they are missing:

.. code-block:: text

   $ agt predict --model model.pth --fasta hg38.fa
   Error: 'agt predict' requires additional dependencies.
   Install them with: pip install alphagenome-pytorch[inference]

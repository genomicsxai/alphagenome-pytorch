Named Outputs
=============

AlphaGenome predicts thousands of genomic tracks — chromatin accessibility,
transcription, histone modifications, TF binding, and more. The named outputs
API lets you work with these tracks by biological meaning (tissue, assay,
ontology) rather than raw channel indices.

.. contents:: On this page
   :local:
   :depth: 2

Setup
-----

.. code-block:: python

   from alphagenome_pytorch import AlphaGenome
   from alphagenome_pytorch.named_outputs import TrackMetadataCatalog

   model = AlphaGenome.from_pretrained("weights.pth", device="cuda")
   catalog = TrackMetadataCatalog.load_builtin()
   model.set_track_metadata_catalog(catalog)

   out = model.predict(dna_onehot, organism_index=0, named_outputs=True)

``out`` is a ``NamedOutputs`` object. Each output head (``out.atac``,
``out.rna_seq``, etc.) is a ``NamedOutputHead`` that holds predictions at one
or more resolutions. Index by resolution to get a ``NamedTrackTensor`` — a
tensor bundled with per-channel metadata:

.. code-block:: python

   out.atac                # NamedOutputHead (all resolutions)
   out.atac[128]           # NamedTrackTensor at 128bp resolution
   out.atac[128].tensor    # the raw torch.Tensor
   out.atac[128].tracks    # tuple of TrackMetadata (one per channel)

Output types and track counts
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 10 12 12 12

   * - Output
     - Resolutions
     - Human tracks
     - Mouse tracks
     - Raw dimension
   * - ``atac``
     - 1bp, 128bp
     - 167
     - 18
     - 256
   * - ``dnase``
     - 1bp, 128bp
     - 305
     - 67
     - 384
   * - ``procap``
     - 1bp, 128bp
     - 12
     - ---
     - 128
   * - ``cage``
     - 1bp, 128bp
     - 546
     - 188
     - 640
   * - ``rna_seq``
     - 1bp, 128bp
     - 667
     - 173
     - 768
   * - ``chip_tf``
     - 128bp
     - 1617
     - 127
     - 1664
   * - ``chip_histone``
     - 128bp
     - 1116
     - 183
     - 1152
   * - ``pair_activations``
     - 128bp
     - 28
     - 8
     - 28

Human/Mouse columns show the number of real (non-padding) tracks. The "Raw
dimension" is the full tensor channel count — both organisms share the same
dimensions, with padding filling the gap. Named outputs strip padding by
default (see :ref:`padding-tracks`).

Track metadata fields
---------------------

Each track carries metadata that you can filter on:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``track_name``
     - Human-readable identifier (e.g. ``"CL:0000084 ATAC-seq"``)
   * - ``strand``
     - ``"+"``, ``"-"``, or ``"."`` (unstranded)
   * - ``ontology_curie``
     - Cell/tissue ontology term (e.g. ``"UBERON:0002107"``)
   * - ``biosample_name``
     - Sample name (e.g. ``"liver"``, ``"HeLa"``)
   * - ``biosample_type``
     - ``"tissue"``, ``"cell_line"``, ``"primary_cell"``, etc.
   * - ``biosample_life_stage``
     - ``"adult"``, ``"embryo"``, etc.
   * - ``assay_title``
     - Assay description (e.g. ``"total RNA-seq"``)
   * - ``gtex_tissue``
     - GTEx tissue name (e.g. ``"Artery_Aorta"``)
   * - ``histone_mark``
     - Histone modification (e.g. ``"H3K27ac"``)
   * - ``transcription_factor``
     - TF name (e.g. ``"CTCF"``)
   * - ``data_source``
     - Data origin (e.g. ``"encode"``, ``"gtex"``)
   * - ``genetically_modified``
     - Whether the sample was genetically modified
   * - ``nonzero_mean``
     - Track mean (used for normalization)

Access fields directly on ``TrackMetadata`` objects:

.. code-block:: python

   track = out.atac[128].tracks[0]

   track.ontology_curie              # Direct attribute access
   track.get('biosample_type')       # Safe access (returns None if missing)
   track.has('genetically_modified') # True if field exists and is not None
   track.to_dict()                   # Serialize to plain dict

Filtering tracks
----------------

The core method is ``.select()``, available on ``NamedTrackTensor``,
``NamedOutputHead``, and ``NamedOutputs``. It returns a new object with only
the matching tracks — both the tensor and metadata are sliced together, so they
stay in sync.

By metadata field
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # By assay
   total_rna = out.rna_seq[1].select(assay_title='total RNA-seq')
   total_rna.tensor   # already sliced to matching channels
   total_rna.tracks   # metadata in sync

   # By histone mark
   h3k27ac = out.chip_histone[128].select(histone_mark='H3K27ac')

   # By tissue
   aorta = out.splice_sites_junction[1].select(gtex_tissue='Artery_Aorta')

   # By ontology
   hepg2_rna = out.rna_seq[1].select(ontology_curie='EFO:0001187')
   liver_rna = out.rna_seq[1].select(ontology_curie='UBERON:0002107')

Multiple conditions
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # All kwargs are AND-ed together
   ctcf_unmodified = out.chip_tf[128].select(
       transcription_factor='CTCF',
       genetically_modified=None,  # field=None matches missing/null values
   )

"Any of" matching
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Pass a list for OR logic within one field
   ctcf_or_foxa1 = out.chip_tf[128].select(
       transcription_factor=['CTCF', 'FOXA1']
   )

Custom predicate
^^^^^^^^^^^^^^^^

.. code-block:: python

   # When kwargs aren't expressive enough, use a predicate function
   liver_related = out.rna_seq[128].select(
       predicate=lambda t: 'liver' in (t.get('biosample_name') or '').lower()
   )

Strand filtering
^^^^^^^^^^^^^^^^

.. code-block:: python

   out.rna_seq[1].select(strand='+')          # positive
   out.rna_seq[1].select(strand='-')          # negative
   out.rna_seq[1].select(strand='.')          # unstranded
   out.rna_seq[1].select(strand=['+', '-'])   # stranded (either)
   out.rna_seq[1].select(strand=['-', '.'])   # non-positive
   out.rna_seq[1].select(strand=['+', '.'])   # non-negative

Masks and indices
-----------------

For loss computation or manual tensor slicing, you can get boolean masks or
integer indices without creating a new ``NamedTrackTensor``:

.. code-block:: python

   # Boolean mask — useful for element-wise loss masking
   mask = out.rna_seq[128].mask(ontology_curie='UBERON:0002107')
   loss = ((preds - targets) ** 2 * mask).mean()

   # Integer indices — useful for gather/index_select
   indices = out.chip_tf[128].indices(transcription_factor='CTCF')
   selected = preds[..., indices]

Resolution-independent queries
------------------------------

Track metadata (names, ontology, biosample, strand, etc.) is the same at all
resolutions — only the tensor's sequence dimension differs. You can query
metadata on the ``NamedOutputHead`` without choosing a resolution:

.. code-block:: python

   head = out.rna_seq
   head.num_tracks        # same at 1bp and 128bp
   head.to_dataframe()    # pandas DataFrame, no resolution needed
   head.indices(strand='+')
   head.mask(strand='+')

Filter at the head level to apply to all resolutions at once:

.. code-block:: python

   plus_strand = out.rna_seq.select(strand='+')
   plus_strand[1].tensor    # 1bp, already filtered
   plus_strand[128].tensor  # 128bp, already filtered

Both orderings produce identical results:

.. code-block:: python

   # These are equivalent
   out.rna_seq.select(strand='+')[128].tensor
   out.rna_seq[128].select(strand='+').tensor

Cross-head filtering
--------------------

Filter across all heads and resolutions at once:

.. code-block:: python

   tissue_tracks = out.select(biosample_type='tissue')
   for (output_name, resolution), ntt in tissue_tracks.items():
       print(f"{output_name}@{resolution}bp: {ntt.num_tracks} tracks")

Variant effect scoring
----------------------

Arithmetic operators (``+``, ``-``, ``*``, ``/``, ``abs``, negation) preserve
metadata from the left operand:

.. code-block:: python

   ref = model.predict(ref_onehot, organism_index=0, named_outputs=True)
   alt = model.predict(alt_onehot, organism_index=0, named_outputs=True)

   # Filter and diff in one expression
   chip_diff = (
       alt.chip_histone[128].select(strand=['-', '.'])
       - ref.chip_histone[128].select(strand=['-', '.'])
   )
   chip_diff.tensor   # the difference tensor
   chip_diff.tracks   # metadata preserved from alt

.. _padding-tracks:

Padding tracks
--------------

The raw model tensors include padding channels so that both organisms share the
same tensor dimensions. For example, the ATAC head always outputs 256 channels,
but only 167 correspond to real human experiments — the remaining 89 are padding
placeholders.

**Named outputs strip padding by default:**

.. code-block:: python

   out = model.predict(dna_onehot, organism_index=0, named_outputs=True)
   out.atac[128].num_tracks  # 167 (real tracks only)

This matches the behavior of the official JAX AlphaGenome API, which also
strips padding before exposing metadata to users.

Keeping padding (for training)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During training you often need the full tensor shape for loss computation with
a mask rather than slicing out channels:

.. code-block:: python

   out = model.predict(
       dna_onehot, organism_index=0,
       named_outputs=True,
       include_padding=True,
   )
   out.atac[128].num_tracks  # 256 (includes padding)

   # Boolean mask: True = real track, False = padding
   mask = out.atac[128].padding_mask()
   loss = ((preds - targets) ** 2 * mask).mean()

   # Resolution-independent mask
   mask = out.atac.padding_mask()

Stripping padding after the fact
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   out.strip_padding()                 # All heads → new NamedOutputs
   out.atac.strip_padding()            # One head → new NamedOutputHead
   out.atac[128].strip_padding()       # One tensor → new NamedTrackTensor

Checking individual tracks
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   for track in out.atac[128].tracks:
       print(track.track_name, track.is_padding)

Loading and building metadata catalogs
--------------------------------------

The built-in catalog ships with the package and contains metadata extracted
from the official JAX AlphaGenome checkpoint:

.. code-block:: python

   from alphagenome_pytorch.named_outputs import TrackMetadataCatalog

   catalog = TrackMetadataCatalog.load_builtin('human')   # or 'mouse'

   # Inspect what's available
   catalog.outputs(organism=0)    # ['atac', 'cage', 'chip_histone', ...]
   catalog.organisms              # [0]

You can also load from files or build programmatically:

.. code-block:: python

   # From parquet / CSV / TSV
   catalog = TrackMetadataCatalog.from_file("my_metadata.parquet")

   # From a pandas DataFrame
   import pandas as pd
   df = pd.DataFrame({
       'track_name': ['sample_A', 'sample_B'],
       'output_type': ['atac', 'atac'],
       'organism': [0, 0],
       'biosample_type': ['tissue', 'cell_line'],
   })
   catalog = TrackMetadataCatalog.from_dataframe(df)

   # Programmatically
   from alphagenome_pytorch.named_outputs import TrackMetadata

   catalog = TrackMetadataCatalog()
   catalog.add_tracks(
       "rna_seq",
       [
           TrackMetadata(0, "rna_seq", 0, "UBERON:0000948 total RNA-seq",
                         extras={"strand": "+", "assay_title": "total RNA-seq"}),
           TrackMetadata(1, "rna_seq", 0, "UBERON:0000948 total RNA-seq",
                         extras={"strand": "-", "assay_title": "total RNA-seq"}),
       ],
       organism=0,
   )

Exporting to pandas
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   df = out.rna_seq[128].to_dataframe()        # One head, one resolution
   df = out.rna_seq.to_dataframe()              # One head (resolution-independent)

Allow empty results
^^^^^^^^^^^^^^^^^^^

By default, ``.select()`` raises ``ValueError`` if no tracks match. Pass
``allow_empty=True`` to get an empty result instead:

.. code-block:: python

   result = out.atac[128].select(biosample_name='nonexistent', allow_empty=True)
   result.num_tracks  # 0

.. _jax-comparison:

Comparison with JAX AlphaGenome
-------------------------------

This section is for users migrating from the JAX ``alphagenome`` /
``alphagenome_research`` packages. 

Loading metadata
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - JAX
     - PyTorch
   * - .. code-block:: python

          model = dna_model.create_from_kaggle('all_folds')
          metadata = model.output_metadata(
              dna_model.Organism.HOMO_SAPIENS
          )
     - .. code-block:: python

          catalog = TrackMetadataCatalog.load_builtin('human')
          model.set_track_metadata_catalog(catalog)

Making predictions
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - JAX
     - PyTorch
   * - .. code-block:: python

          predictions = model.predict_interval(
              interval,
              requested_outputs={
                  dna_model.OutputType.RNA_SEQ,
              },
              ontology_terms=['EFO:0001187'],
          )
          predictions.rna_seq.metadata
     - .. code-block:: python

          out = model.predict(
              dna_onehot, organism_index=0,
              named_outputs=True,
          )
          out.rna_seq[1].to_dataframe()

Filtering
^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - JAX
     - PyTorch
   * - .. code-block:: python

          # Boolean mask with pandas
          predictions.rna_seq.filter_tracks(
              (predictions.rna_seq.metadata[
                  'Assay title'
              ] == 'total RNA-seq').values
          )

          # Multiple conditions
          predictions.chip_tf.filter_tracks(
              (
                  (metadata['transcription_factor']
                   == 'CTCF')
                  & (metadata[
                      'genetically_modified'
                  ].isnull())
              ).values
          )
     - .. code-block:: python

          # Keyword arguments
          out.rna_seq[1].select(
              assay_title='total RNA-seq'
          )

          # Multiple conditions
          out.chip_tf[128].select(
              transcription_factor='CTCF',
              genetically_modified=None,
          )

Strand filtering
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - JAX
     - PyTorch
   * - .. code-block:: python

          # 6 dedicated methods
          predictions.rna_seq \
              .filter_to_positive_strand()
          predictions.rna_seq \
              .filter_to_nonpositive_strand()
          predictions.splice_junctions \
              .filter_to_strand('+')
     - .. code-block:: python

          # All via select()
          out.rna_seq[1].select(strand='+')
          out.rna_seq[1].select(
              strand=['-', '.']
          )

Padding
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - JAX
     - PyTorch
   * - .. code-block:: python

          # Manual boolean mask
          padding = metadata.padding
          mask = ~padding[OutputType.ATAC]

          # Or via create_track_masks()
          masks = metadata_lib \
              .create_track_masks(
                  metadata,
                  requested_outputs={...},
                  requested_ontologies=None,
              )
     - .. code-block:: python

          # Stripped by default
          out.atac[128].num_tracks  # 167

          # Or keep + mask
          out = model.predict(
              dna, 0,
              named_outputs=True,
              include_padding=True,
          )
          mask = out.atac.padding_mask()
          track.is_padding  # per track

Feature comparison table
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 30 30

   * - Feature
     - JAX
     - PyTorch
   * - Load metadata
     - ``model.output_metadata()``
     - ``TrackMetadataCatalog.load_builtin()``
   * - Filter by field
     - ``.filter_tracks(bool_mask)``
     - ``.select(**criteria)``
   * - Filter null/missing
     - ``metadata['col'].isnull()``
     - ``.select(field=None)``
   * - Access metadata field
     - ``metadata['field']``
     - ``track.field``
   * - Safe field access
     - ---
     - ``track.get('field', default)``
   * - Check field exists
     - ---
     - ``track.has('field')``
   * - Strand filtering
     - ``.filter_to_strand('+')``
     - ``.select(strand='+')``
   * - Tissue filtering
     - ``.filter_by_tissue(...)``
     - ``.select(gtex_tissue=...)``
   * - Get indices
     - manual numpy
     - ``.indices()``
   * - Get boolean mask
     - manual numpy
     - ``.mask()``
   * - Padding detection
     - ``name.str.lower() == 'padding'``
     - ``track.is_padding``
   * - Strip padding
     - ``metadata[~metadata['padding']]``
     - auto / ``.strip_padding()``
   * - Padding mask
     - ``~metadata.padding[type]``
     - ``.padding_mask()``
   * - To DataFrame
     - ``.metadata`` property
     - ``.to_dataframe()``
   * - Arithmetic
     - direct on objects
     - direct on objects
   * - Cross-head filtering
     - ---
     - ``.select(**criteria)``
   * - Allow empty results
     - ---
     - ``.select(allow_empty=True)``
   * - Load from DataFrame
     - ---
     - ``.from_dataframe(df)``

Design notes
^^^^^^^^^^^^

**Why metadata and tensor are bundled** (``NamedTrackTensor``): After any
``.select()`` call, the returned tensor and metadata are guaranteed to be
aligned — no manual index tracking needed.

**Why metadata is resolution-independent** (``NamedOutputHead``): Track metadata
doesn't change between 1bp and 128bp — only the sequence dimension differs.
``NamedOutputHead`` lets you query metadata and filter without choosing a
resolution first.

"""JAX vs PyTorch metadata comparison for named outputs.

Validates that the built-in PyTorch TrackMetadataCatalog (loaded from parquet)
matches the JAX reference OutputMetadata (loaded from textproto via
alphagenome_research). This ensures the extract_track_metadata.py pipeline
produced correct metadata and that TrackMetadataCatalog loads it faithfully.

Note on padding:
    For most output types, both organisms are padded to the same head output
    dimension in the JAX metadata itself (rows named "Padding"). However,
    SPLICE_JUNCTIONS is an exception: the JAX metadata has 367 tracks for human
    but only 90 for mouse, with no padding rows. The model's
    SpliceSitesJunctionHead uses a shared num_tissues=367 dimension and masks
    unused mouse channels at runtime. Our extract_track_metadata.py script pads
    the mouse parquet from 90 → 367 with placeholder rows to match the tensor
    shape. The tests below therefore allow len(pt) >= len(jax) and compare only
    the first len(jax) tracks.
"""

import pytest

# JAX output type enum name → PyTorch output_type string
# (must match the mapping used in scripts/extract_track_metadata.py)
_OUTPUT_TYPE_MAP = {
    "ATAC": "atac",
    "DNASE": "dnase",
    "PROCAP": "procap",
    "CAGE": "cage",
    "RNA_SEQ": "rna_seq",
    "CHIP_TF": "chip_tf",
    "CHIP_HISTONE": "chip_histone",
    "CONTACT_MAPS": "pair_activations",
    "SPLICE_SITES": "splice_sites_classification",
    "SPLICE_SITE_USAGE": "splice_sites_usage",
    "SPLICE_JUNCTIONS": "splice_sites_junction",
}

# JAX DataFrame column → PyTorch TrackMetadata field or extras key
# (from scripts/extract_track_metadata.py COLUMN_MAPPING)
_COLUMN_MAPPING = {
    "name": "track_name",  # core field
    "strand": "strand",
    "ontology_curie": "ontology_curie",
    "biosample_name": "biosample_name",
    "biosample_type": "biosample_type",
    "biosample_life_stage": "biosample_life_stage",
    "Assay title": "assay_title",
    "gtex_tissue": "gtex_tissue",
    "histone_mark": "histone_mark",
    "transcription_factor": "transcription_factor",
    "data_source": "data_source",
    "endedness": "endedness",
    "genetically_modified": "genetically_modified",
    "nonzero_mean": "nonzero_mean",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def jax_modules(jax_available, alphagenome_research_available):
    """Import JAX metadata modules (skipped if unavailable)."""
    from alphagenome.models import dna_model
    from alphagenome.models import dna_output
    from alphagenome_research.model.metadata import metadata as metadata_lib
    return dna_model, dna_output, metadata_lib


@pytest.fixture(scope="module")
def jax_metadata_human(jax_modules):
    dna_model, _, metadata_lib = jax_modules
    return metadata_lib.load(dna_model.Organism.HOMO_SAPIENS)


@pytest.fixture(scope="module")
def jax_metadata_mouse(jax_modules):
    dna_model, _, metadata_lib = jax_modules
    return metadata_lib.load(dna_model.Organism.MUS_MUSCULUS)


@pytest.fixture(scope="module")
def pt_catalog_human():
    from alphagenome_pytorch.named_outputs import TrackMetadataCatalog
    return TrackMetadataCatalog.load_builtin("human")


@pytest.fixture(scope="module")
def pt_catalog_mouse():
    from alphagenome_pytorch.named_outputs import TrackMetadataCatalog
    return TrackMetadataCatalog.load_builtin("mouse")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_jax_df(jax_metadata, jax_modules, output_type_name: str):
    """Return the JAX DataFrame for a given OutputType name, or None."""
    _, dna_output, _ = jax_modules
    output_type = getattr(dna_output.OutputType, output_type_name)
    return jax_metadata.get(output_type)


def _get_pt_tracks(pt_catalog, output_type_name: str, organism: int = 0):
    """Return PyTorch TrackMetadata tuple for the mapped output_type string."""
    pt_output_name = _OUTPUT_TYPE_MAP[output_type_name]
    return pt_catalog.get_tracks(pt_output_name, organism=organism)


def _pt_field(track, field_name: str):
    """Get a field from a PyTorch TrackMetadata (core attr or extras)."""
    if hasattr(track, field_name) and field_name != "extras":
        return getattr(track, field_name)
    return track.extras.get(field_name)


def _clean_value(val):
    """Normalize None/NaN/empty to None for comparison."""
    if val is None:
        return None
    s = str(val).strip()
    if s.lower() in ("", "nan", "none", "null"):
        return None
    return s


# ---------------------------------------------------------------------------
# Tests — parametrized over output types
# ---------------------------------------------------------------------------

_OUTPUT_TYPES = list(_OUTPUT_TYPE_MAP.keys())


@pytest.mark.parametrize("output_type", _OUTPUT_TYPES)
class TestHumanMetadata:
    """Compare JAX and PyTorch metadata for human organism."""

    def test_track_count_matches(
        self, output_type, jax_metadata_human, jax_modules, pt_catalog_human
    ):
        jax_df = _get_jax_df(jax_metadata_human, jax_modules, output_type)
        pt_tracks = _get_pt_tracks(pt_catalog_human, output_type, organism=0)
        if jax_df is None:
            assert len(pt_tracks) == 0, (
                f"JAX has no metadata for {output_type}, "
                f"but PyTorch has {len(pt_tracks)} tracks"
            )
            return
        # PyTorch may have more tracks than JAX due to padding added by
        # extract_track_metadata.py (see module docstring for details).
        assert len(pt_tracks) >= len(jax_df), (
            f"{output_type}: JAX has {len(jax_df)} tracks, "
            f"but PyTorch only has {len(pt_tracks)}"
        )

    def test_track_names_match(
        self, output_type, jax_metadata_human, jax_modules, pt_catalog_human
    ):
        jax_df = _get_jax_df(jax_metadata_human, jax_modules, output_type)
        pt_tracks = _get_pt_tracks(pt_catalog_human, output_type, organism=0)
        if jax_df is None or len(pt_tracks) == 0:
            pytest.skip(f"No metadata for {output_type}")

        jax_names = jax_df["name"].tolist()
        pt_names = [t.track_name for t in pt_tracks]

        count = min(len(jax_names), len(pt_names))
        for i in range(count):
            assert pt_names[i] == jax_names[i], (
                f"{output_type} track {i}: "
                f"JAX name={jax_names[i]!r}, PyTorch name={pt_names[i]!r}"
            )

    def test_strand_values_match(
        self, output_type, jax_metadata_human, jax_modules, pt_catalog_human
    ):
        jax_df = _get_jax_df(jax_metadata_human, jax_modules, output_type)
        pt_tracks = _get_pt_tracks(pt_catalog_human, output_type, organism=0)
        if jax_df is None or len(pt_tracks) == 0:
            pytest.skip(f"No metadata for {output_type}")
        if "strand" not in jax_df.columns:
            pytest.skip(f"No strand column in JAX metadata for {output_type}")

        jax_strands = jax_df["strand"].tolist()
        pt_strands = [_pt_field(t, "strand") for t in pt_tracks]

        count = min(len(jax_strands), len(pt_strands))
        for i in range(count):
            assert _clean_value(pt_strands[i]) == _clean_value(jax_strands[i]), (
                f"{output_type} track {i}: "
                f"JAX strand={jax_strands[i]!r}, PyTorch strand={pt_strands[i]!r}"
            )

    def test_ontology_curie_matches(
        self, output_type, jax_metadata_human, jax_modules, pt_catalog_human
    ):
        jax_df = _get_jax_df(jax_metadata_human, jax_modules, output_type)
        pt_tracks = _get_pt_tracks(pt_catalog_human, output_type, organism=0)
        if jax_df is None or len(pt_tracks) == 0:
            pytest.skip(f"No metadata for {output_type}")
        if "ontology_curie" not in jax_df.columns:
            pytest.skip(f"No ontology_curie column for {output_type}")

        jax_values = jax_df["ontology_curie"].tolist()
        pt_values = [_pt_field(t, "ontology_curie") for t in pt_tracks]

        count = min(len(jax_values), len(pt_values))
        for i in range(count):
            assert _clean_value(pt_values[i]) == _clean_value(jax_values[i]), (
                f"{output_type} track {i}: "
                f"JAX ontology={jax_values[i]!r}, PyTorch ontology={pt_values[i]!r}"
            )

    def test_biosample_name_matches(
        self, output_type, jax_metadata_human, jax_modules, pt_catalog_human
    ):
        jax_df = _get_jax_df(jax_metadata_human, jax_modules, output_type)
        pt_tracks = _get_pt_tracks(pt_catalog_human, output_type, organism=0)
        if jax_df is None or len(pt_tracks) == 0:
            pytest.skip(f"No metadata for {output_type}")
        if "biosample_name" not in jax_df.columns:
            pytest.skip(f"No biosample_name column for {output_type}")

        jax_values = jax_df["biosample_name"].tolist()
        pt_values = [_pt_field(t, "biosample_name") for t in pt_tracks]

        count = min(len(jax_values), len(pt_values))
        for i in range(count):
            assert _clean_value(pt_values[i]) == _clean_value(jax_values[i]), (
                f"{output_type} track {i}: "
                f"JAX biosample={jax_values[i]!r}, PyTorch biosample={pt_values[i]!r}"
            )


@pytest.mark.parametrize("output_type", _OUTPUT_TYPES)
class TestMouseMetadata:
    """Compare JAX and PyTorch metadata for mouse organism."""

    def test_track_count_matches(
        self, output_type, jax_metadata_mouse, jax_modules, pt_catalog_mouse
    ):
        jax_df = _get_jax_df(jax_metadata_mouse, jax_modules, output_type)
        pt_tracks = _get_pt_tracks(pt_catalog_mouse, output_type, organism=1)
        if jax_df is None:
            assert len(pt_tracks) == 0, (
                f"JAX has no metadata for {output_type}, "
                f"but PyTorch has {len(pt_tracks)} tracks"
            )
            return
        # PyTorch may have more tracks than JAX due to padding added by
        # extract_track_metadata.py (e.g. SPLICE_JUNCTIONS mouse: JAX has 90
        # real tracks, PyTorch parquet has 367 to match the head dimension).
        assert len(pt_tracks) >= len(jax_df), (
            f"{output_type}: JAX has {len(jax_df)} tracks, "
            f"but PyTorch only has {len(pt_tracks)}"
        )

    def test_track_names_match(
        self, output_type, jax_metadata_mouse, jax_modules, pt_catalog_mouse
    ):
        jax_df = _get_jax_df(jax_metadata_mouse, jax_modules, output_type)
        pt_tracks = _get_pt_tracks(pt_catalog_mouse, output_type, organism=1)
        if jax_df is None or len(pt_tracks) == 0:
            pytest.skip(f"No metadata for {output_type}")

        jax_names = jax_df["name"].tolist()
        pt_names = [t.track_name for t in pt_tracks]

        count = min(len(jax_names), len(pt_names))
        for i in range(count):
            assert pt_names[i] == jax_names[i], (
                f"{output_type} track {i}: "
                f"JAX name={jax_names[i]!r}, PyTorch name={pt_names[i]!r}"
            )

    def test_strand_values_match(
        self, output_type, jax_metadata_mouse, jax_modules, pt_catalog_mouse
    ):
        jax_df = _get_jax_df(jax_metadata_mouse, jax_modules, output_type)
        pt_tracks = _get_pt_tracks(pt_catalog_mouse, output_type, organism=1)
        if jax_df is None or len(pt_tracks) == 0:
            pytest.skip(f"No metadata for {output_type}")
        if "strand" not in jax_df.columns:
            pytest.skip(f"No strand column for {output_type}")

        jax_strands = jax_df["strand"].tolist()
        pt_strands = [_pt_field(t, "strand") for t in pt_tracks]

        count = min(len(jax_strands), len(pt_strands))
        for i in range(count):
            assert _clean_value(pt_strands[i]) == _clean_value(jax_strands[i]), (
                f"{output_type} track {i}: "
                f"JAX strand={jax_strands[i]!r}, PyTorch strand={pt_strands[i]!r}"
            )

    def test_ontology_curie_matches(
        self, output_type, jax_metadata_mouse, jax_modules, pt_catalog_mouse
    ):
        jax_df = _get_jax_df(jax_metadata_mouse, jax_modules, output_type)
        pt_tracks = _get_pt_tracks(pt_catalog_mouse, output_type, organism=1)
        if jax_df is None or len(pt_tracks) == 0:
            pytest.skip(f"No metadata for {output_type}")
        if "ontology_curie" not in jax_df.columns:
            pytest.skip(f"No ontology_curie column for {output_type}")

        jax_values = jax_df["ontology_curie"].tolist()
        pt_values = [_pt_field(t, "ontology_curie") for t in pt_tracks]

        count = min(len(jax_values), len(pt_values))
        for i in range(count):
            assert _clean_value(pt_values[i]) == _clean_value(jax_values[i]), (
                f"{output_type} track {i}: "
                f"JAX ontology={jax_values[i]!r}, PyTorch ontology={pt_values[i]!r}"
            )

    def test_biosample_name_matches(
        self, output_type, jax_metadata_mouse, jax_modules, pt_catalog_mouse
    ):
        jax_df = _get_jax_df(jax_metadata_mouse, jax_modules, output_type)
        pt_tracks = _get_pt_tracks(pt_catalog_mouse, output_type, organism=1)
        if jax_df is None or len(pt_tracks) == 0:
            pytest.skip(f"No metadata for {output_type}")
        if "biosample_name" not in jax_df.columns:
            pytest.skip(f"No biosample_name column for {output_type}")

        jax_values = jax_df["biosample_name"].tolist()
        pt_values = [_pt_field(t, "biosample_name") for t in pt_tracks]

        count = min(len(jax_values), len(pt_values))
        for i in range(count):
            assert _clean_value(pt_values[i]) == _clean_value(jax_values[i]), (
                f"{output_type} track {i}: "
                f"JAX biosample={jax_values[i]!r}, PyTorch biosample={pt_values[i]!r}"
            )


class TestOutputTypeCoverage:
    """Verify all JAX output types are covered in PyTorch metadata."""

    def test_all_jax_output_types_present_human(
        self, jax_metadata_human, jax_modules, pt_catalog_human
    ):
        _, dna_output, _ = jax_modules
        missing = []
        for output_type in dna_output.OutputType:
            jax_df = jax_metadata_human.get(output_type)
            if jax_df is not None and len(jax_df) > 0:
                pt_name = _OUTPUT_TYPE_MAP.get(output_type.name)
                if pt_name is None:
                    missing.append(f"{output_type.name} (no mapping)")
                    continue
                pt_tracks = pt_catalog_human.get_tracks(pt_name, organism=0)
                if len(pt_tracks) == 0:
                    missing.append(f"{output_type.name} (mapped to '{pt_name}', 0 tracks)")
        assert not missing, f"Missing PyTorch metadata for human: {missing}"

    def test_all_jax_output_types_present_mouse(
        self, jax_metadata_mouse, jax_modules, pt_catalog_mouse
    ):
        _, dna_output, _ = jax_modules
        missing = []
        for output_type in dna_output.OutputType:
            jax_df = jax_metadata_mouse.get(output_type)
            if jax_df is not None and len(jax_df) > 0:
                pt_name = _OUTPUT_TYPE_MAP.get(output_type.name)
                if pt_name is None:
                    missing.append(f"{output_type.name} (no mapping)")
                    continue
                pt_tracks = pt_catalog_mouse.get_tracks(pt_name, organism=1)
                if len(pt_tracks) == 0:
                    missing.append(f"{output_type.name} (mapped to '{pt_name}', 0 tracks)")
        assert not missing, f"Missing PyTorch metadata for mouse: {missing}"

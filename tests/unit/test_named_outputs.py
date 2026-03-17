"""Unit tests for metadata-aware named outputs."""

import pytest
import torch

from alphagenome_pytorch.named_outputs import (
    NamedOutputs,
    TrackMetadata,
    TrackMetadataCatalog,
)


@pytest.mark.unit
def test_named_outputs_where_filters_by_extras():
    """Filter tracks by fields stored in extras (ontology_curie, etc.)."""
    rows = [
        {
            "organism": "human",
            "output_type": "atac",
            "track_name": "liver_track",
            "ontology_curie": "UBERON:0002107",
        },
        {
            "organism": "human",
            "output_type": "atac",
            "track_name": "brain_track",
            "ontology_curie": "UBERON:0000955",
        },
        {
            "organism": "human",
            "output_type": "atac",
            "track_name": "liver_track_2",
            "ontology_curie": "UBERON:0002107",
        },
    ]
    catalog = TrackMetadataCatalog.from_rows(rows)

    outputs = {"atac": {128: torch.randn(2, 8, 3)}}
    named = NamedOutputs.from_raw(outputs, organism=0, catalog=catalog)

    selected = named.atac[128].select(ontology_curie="UBERON:0002107")
    assert selected.tensor.shape[-1] == 2
    assert [track.track_name for track in selected.tracks] == ["liver_track", "liver_track_2"]


@pytest.mark.unit
def test_named_outputs_where_filters_by_core_attribute():
    """Filter tracks by core attributes (track_name, organism)."""
    rows = [
        {"organism": "human", "output_type": "atac", "track_name": "liver"},
        {"organism": "human", "output_type": "atac", "track_name": "brain"},
        {"organism": "human", "output_type": "atac", "track_name": "liver"},
    ]
    catalog = TrackMetadataCatalog.from_rows(rows)

    outputs = {"atac": {128: torch.randn(2, 8, 3)}}
    named = NamedOutputs.from_raw(outputs, organism=0, catalog=catalog)

    selected = named.atac[128].select(track_name="liver")
    assert selected.tensor.shape[-1] == 2


@pytest.mark.unit
def test_named_outputs_where_with_predicate():
    """Filter tracks using a custom predicate function."""
    rows = [
        {"organism": "human", "output_type": "atac", "track_name": "liver_sample_1"},
        {"organism": "human", "output_type": "atac", "track_name": "brain_sample"},
        {"organism": "human", "output_type": "atac", "track_name": "liver_sample_2"},
    ]
    catalog = TrackMetadataCatalog.from_rows(rows)

    outputs = {"atac": {128: torch.randn(2, 8, 3)}}
    named = NamedOutputs.from_raw(outputs, organism=0, catalog=catalog)

    selected = named.atac[128].select(predicate=lambda t: "liver" in t.track_name)
    assert selected.tensor.shape[-1] == 2
    assert all("liver" in t.track_name for t in selected.tracks)


@pytest.mark.unit
def test_named_outputs_where_with_collection():
    """Filter tracks using a collection of values (in-matching)."""
    rows = [
        {"organism": "human", "output_type": "atac", "track_name": "a", "biosample_type": "tissue"},
        {"organism": "human", "output_type": "atac", "track_name": "b", "biosample_type": "cell_line"},
        {"organism": "human", "output_type": "atac", "track_name": "c", "biosample_type": "primary_cell"},
    ]
    catalog = TrackMetadataCatalog.from_rows(rows)

    outputs = {"atac": {128: torch.randn(2, 8, 3)}}
    named = NamedOutputs.from_raw(outputs, organism=0, catalog=catalog)

    selected = named.atac[128].select(biosample_type=["tissue", "primary_cell"])
    assert selected.tensor.shape[-1] == 2
    assert [t.track_name for t in selected.tracks] == ["a", "c"]


@pytest.mark.unit
def test_named_outputs_without_catalog_uses_placeholders():
    """Without a catalog, placeholder track names are generated."""
    outputs = {"atac": {128: torch.randn(1, 4, 2)}}
    named = NamedOutputs.from_raw(outputs, organism=0, catalog=None)

    tracks = named.atac[128].tracks
    assert tracks[0].track_name == "track_0"
    assert tracks[1].track_name == "track_1"


@pytest.mark.unit
def test_named_outputs_strict_metadata_requires_catalog_entries():
    """strict_metadata=True raises when catalog has no matching entries."""
    outputs = {"atac": {128: torch.randn(1, 4, 2)}}
    catalog = TrackMetadataCatalog()

    with pytest.raises(KeyError, match="No metadata found"):
        NamedOutputs.from_raw(outputs, organism=0, catalog=catalog, strict_metadata=True)


@pytest.mark.unit
def test_catalog_from_csv_normalizes_output_alias(tmp_path):
    """Output name aliases (contact_maps -> pair_activations) are normalized."""
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(
        "organism,output_type,track_name\n"
        "human,contact_maps,cm_0\n"
        "human,contact_maps,cm_1\n",
        encoding="utf-8",
    )

    catalog = TrackMetadataCatalog.from_file(csv_path)
    tracks = catalog.get_tracks("pair_activations", organism=0, num_tracks=2, strict=True)

    assert len(tracks) == 2
    assert tracks[0].output_name == "pair_activations"
    assert tracks[1].track_name == "cm_1"


@pytest.mark.unit
def test_track_metadata_extras_contain_non_core_fields():
    """Non-core fields are stored in extras dict."""
    rows = [
        {
            "organism": "human",
            "output_type": "atac",
            "track_name": "sample_1",
            "ontology_curie": "UBERON:0002107",
            "biosample_type": "tissue",
            "data_source": "encode",
            "custom_field": "custom_value",
        },
    ]
    catalog = TrackMetadataCatalog.from_rows(rows)
    tracks = catalog.get_tracks("atac", organism=0)

    assert len(tracks) == 1
    track = tracks[0]
    assert track.track_name == "sample_1"
    assert track.extras["ontology_curie"] == "UBERON:0002107"
    assert track.extras["biosample_type"] == "tissue"
    assert track.extras["data_source"] == "encode"
    assert track.extras["custom_field"] == "custom_value"


@pytest.mark.unit
def test_named_outputs_len():
    """NamedOutputs supports len()."""
    outputs = {"atac": {128: torch.randn(1, 4, 2)}, "dnase": {128: torch.randn(1, 4, 3)}}
    named = NamedOutputs.from_raw(outputs, organism=0, catalog=None)
    assert len(named) == 2


@pytest.mark.unit
def test_named_outputs_dict_interface():
    """NamedOutputs supports dict-like access."""
    outputs = {"atac": {128: torch.randn(1, 4, 2)}}
    named = NamedOutputs.from_raw(outputs, organism=0, catalog=None)

    assert "atac" in named
    assert list(named.keys()) == ["atac"]
    assert named.heads() == ["atac"]
    assert named.atac[128].num_tracks == 2


@pytest.mark.unit
def test_named_output_head_tracks_shared_across_resolutions():
    """NamedOutputHead exposes shared metadata without choosing resolution."""
    rows = [
        {"organism": "human", "output_type": "atac", "track_name": "liver", "strand": "+"},
        {"organism": "human", "output_type": "atac", "track_name": "brain", "strand": "-"},
    ]
    catalog = TrackMetadataCatalog.from_rows(rows)

    outputs = {"atac": {1: torch.randn(1, 128, 2), 128: torch.randn(1, 1, 2)}}
    named = NamedOutputs.from_raw(outputs, organism=0, catalog=catalog)

    head = named.atac
    assert head.num_tracks == 2
    assert [t.track_name for t in head.tracks] == ["liver", "brain"]


@pytest.mark.unit
def test_named_output_head_select_filters_all_resolutions():
    """NamedOutputHead.select() filters tensors at all resolutions."""
    rows = [
        {"organism": "human", "output_type": "atac", "track_name": "liver", "strand": "+"},
        {"organism": "human", "output_type": "atac", "track_name": "brain", "strand": "-"},
        {"organism": "human", "output_type": "atac", "track_name": "kidney", "strand": "+"},
    ]
    catalog = TrackMetadataCatalog.from_rows(rows)

    outputs = {"atac": {1: torch.randn(1, 128, 3), 128: torch.randn(1, 1, 3)}}
    named = NamedOutputs.from_raw(outputs, organism=0, catalog=catalog)

    filtered = named.atac.select(strand="+")
    assert filtered.num_tracks == 2
    assert filtered[1].tensor.shape[-1] == 2
    assert filtered[128].tensor.shape[-1] == 2
    assert [t.track_name for t in filtered.tracks] == ["liver", "kidney"]


@pytest.mark.unit
def test_named_output_head_select_order_independence():
    """head.select()[res] and head[res].select() give equivalent results."""
    rows = [
        {"organism": "human", "output_type": "atac", "track_name": "a", "strand": "+"},
        {"organism": "human", "output_type": "atac", "track_name": "b", "strand": "-"},
        {"organism": "human", "output_type": "atac", "track_name": "c", "strand": "+"},
    ]
    catalog = TrackMetadataCatalog.from_rows(rows)

    tensor_1bp = torch.randn(1, 128, 3)
    tensor_128bp = torch.randn(1, 1, 3)
    outputs = {"atac": {1: tensor_1bp, 128: tensor_128bp}}
    named = NamedOutputs.from_raw(outputs, organism=0, catalog=catalog)

    # Order 1: select first, then pick resolution
    via_head = named.atac.select(strand="+")[128]
    # Order 2: pick resolution first, then select
    via_tensor = named.atac[128].select(strand="+")

    assert via_head.num_tracks == via_tensor.num_tracks == 2
    assert [t.track_name for t in via_head.tracks] == [t.track_name for t in via_tensor.tracks]
    assert torch.equal(via_head.tensor, via_tensor.tensor)


@pytest.mark.unit
def test_named_output_head_to_dataframe():
    """NamedOutputHead.to_dataframe() returns a DataFrame without choosing resolution."""
    pd = pytest.importorskip("pandas")
    rows = [
        {"organism": "human", "output_type": "atac", "track_name": "liver", "strand": "+"},
        {"organism": "human", "output_type": "atac", "track_name": "brain", "strand": "-"},
    ]
    catalog = TrackMetadataCatalog.from_rows(rows)

    outputs = {"atac": {1: torch.randn(1, 128, 2), 128: torch.randn(1, 1, 2)}}
    named = NamedOutputs.from_raw(outputs, organism=0, catalog=catalog)

    df = named.atac.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df["track_name"]) == ["liver", "brain"]


@pytest.mark.unit
def test_named_output_head_indices_and_mask():
    """NamedOutputHead.indices() and .mask() work without choosing resolution."""
    rows = [
        {"organism": "human", "output_type": "atac", "track_name": "a", "strand": "+"},
        {"organism": "human", "output_type": "atac", "track_name": "b", "strand": "-"},
        {"organism": "human", "output_type": "atac", "track_name": "c", "strand": "+"},
    ]
    catalog = TrackMetadataCatalog.from_rows(rows)

    outputs = {"atac": {1: torch.randn(1, 128, 3), 128: torch.randn(1, 1, 3)}}
    named = NamedOutputs.from_raw(outputs, organism=0, catalog=catalog)

    indices = named.atac.indices(strand="+")
    assert indices == [0, 2]

    mask = named.atac.mask(strand="+")
    assert mask.tolist() == [True, False, True]


@pytest.mark.unit
def test_named_output_head_select_allow_empty():
    """NamedOutputHead.select(allow_empty=True) returns empty head."""
    rows = [
        {"organism": "human", "output_type": "atac", "track_name": "a", "strand": "+"},
    ]
    catalog = TrackMetadataCatalog.from_rows(rows)

    outputs = {"atac": {128: torch.randn(1, 1, 1)}}
    named = NamedOutputs.from_raw(outputs, organism=0, catalog=catalog)

    filtered = named.atac.select(strand="-", allow_empty=True)
    assert filtered.num_tracks == 0
    assert filtered[128].tensor.shape[-1] == 0


@pytest.mark.unit
def test_named_outputs_cross_head_select():
    """NamedOutputs.select() filters across all heads and resolutions."""
    rows = [
        {"organism": "human", "output_type": "atac", "track_name": "liver", "ontology_curie": "UBERON:0002107"},
        {"organism": "human", "output_type": "atac", "track_name": "brain", "ontology_curie": "UBERON:0000955"},
        {"organism": "human", "output_type": "dnase", "track_name": "liver_dnase", "ontology_curie": "UBERON:0002107"},
        {"organism": "human", "output_type": "dnase", "track_name": "heart_dnase", "ontology_curie": "UBERON:0000948"},
    ]
    catalog = TrackMetadataCatalog.from_rows(rows)

    outputs = {
        "atac": {128: torch.randn(1, 1, 2)},
        "dnase": {128: torch.randn(1, 1, 2)},
    }
    named = NamedOutputs.from_raw(outputs, organism=0, catalog=catalog)

    result = named.select(ontology_curie="UBERON:0002107")
    # Both heads should have matches
    assert ("atac", 128) in result
    assert ("dnase", 128) in result
    assert result[("atac", 128)].num_tracks == 1
    assert result[("dnase", 128)].num_tracks == 1
    assert result[("atac", 128)].tracks[0].track_name == "liver"
    assert result[("dnase", 128)].tracks[0].track_name == "liver_dnase"


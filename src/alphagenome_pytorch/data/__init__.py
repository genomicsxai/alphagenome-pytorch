"""Built-in track metadata for AlphaGenome.

This directory contains pre-extracted track metadata from the AlphaGenome model:
- track_metadata_human.parquet: Human (Homo sapiens) track metadata
- track_metadata_mouse.parquet: Mouse (Mus musculus) track metadata

Usage:
    from alphagenome_pytorch.named_outputs import TrackMetadataCatalog

    # Load built-in metadata
    catalog = TrackMetadataCatalog.load_builtin("human")

    # Or load from file path
    catalog = TrackMetadataCatalog.from_file("path/to/metadata.parquet")

To regenerate these files (requires alphagenome and alphagenome_research):
    python scripts/extract_track_metadata.py
"""

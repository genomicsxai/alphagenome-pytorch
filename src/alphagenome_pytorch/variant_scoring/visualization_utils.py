# Visualization imports and helpers
from alphagenome.data import track_data as jax_track_data
from alphagenome.data import genome as jax_genome
from alphagenome.data import gene_annotation, transcript
from alphagenome.visualization import plot_components
from .types import OutputType
import pandas as pd
import numpy as np
import torch

# Cache for transcript extractors
_transcript_extractor_cache = {}


def get_transcript_extractors_from_df(gtf_df):
    """Build transcript extractors from a GTF-style DataFrame."""
    # Filter to protein-coding genes and highly supported transcripts
    gtf_transcript = gene_annotation.filter_transcript_support_level(
        gene_annotation.filter_protein_coding(gtf_df), ['1']
    )
    transcript_extractor = transcript.TranscriptExtractor(gtf_transcript)
    
    # Longest transcript per gene
    gtf_longest = gene_annotation.filter_to_longest_transcript(gtf_transcript)
    longest_transcript_extractor = transcript.TranscriptExtractor(gtf_longest)
    
    return transcript_extractor, longest_transcript_extractor


def get_transcript_extractors(scoring_model, organism='human'):
    """Get transcript extractors, using scoring_model's annotation if available."""
    cache_key = id(scoring_model) if scoring_model._gene_annotation else organism
    
    if cache_key in _transcript_extractor_cache:
        return _transcript_extractor_cache[cache_key]
    
    # Use the scoring model's gene annotation if available
    if scoring_model._gene_annotation is not None:
        gtf_df = scoring_model._gene_annotation.df
        extractors = get_transcript_extractors_from_df(gtf_df)
    else:
        raise ValueError(
            f"scoring_model._gene_annotation is None. Automatic download for {organism} "
            "is not supported."
        )
    
    _transcript_extractor_cache[cache_key] = extractors
    return extractors


def track_metadata_to_df(track_metadata_list, output_type_label=None):
    """Convert list[TrackMetadata] to DataFrame for JAX TrackData."""
    # Handle both TrackMetadata objects and ad-hoc objects/dicts
    data = []
    for m in track_metadata_list:
        item = {
            'name': getattr(m, 'track_name', ''),
            'strand': getattr(m, 'track_strand', '.'),
            'biosample_name': getattr(m, 'biosample_name', ''),
            'ontology_curie': getattr(m, 'ontology_curie', ''),
            'assay_title': getattr(m, 'assay_title', ''),
            'transcription_factor': getattr(m, 'transcription_factor', ''),
            'histone_mark': getattr(m, 'histone_mark', ''),
            'output_type': output_type_label or '',
        }
        data.append(item)
    return pd.DataFrame(data)


def pytorch_to_track_data(predictions, track_metadata_list, interval, resolution=1, output_type_label=None):
    """Convert PyTorch predictions + TrackMetadata list to JAX TrackData."""
    if hasattr(predictions, 'cpu'):
        # .float() handles bfloat16 -> float32 conversion for numpy compatibility
        values = predictions.squeeze(0).float().cpu().numpy()
    else:
        values = np.asarray(predictions).squeeze(0)
    
    meta_df = track_metadata_to_df(track_metadata_list, output_type_label)
    jax_interval = jax_genome.Interval(
        chromosome=interval.chromosome,
        start=interval.start,
        end=interval.end,
    )
    
    return jax_track_data.TrackData(
        values=values,
        metadata=meta_df,
        resolution=resolution,
        interval=jax_interval,
    )


def extract_predictions(outputs, output_type, preferred_resolution=None):
    """Extract predictions tensor from model outputs, handling different output structures.
    
    Args:
        outputs: Model outputs dict
        output_type: OutputType enum
        preferred_resolution: Preferred resolution (1 or 128). If None, prefers 128bp when available.
    
    Returns:
        (predictions_tensor, resolution) tuple
    """
    output_key = output_type.value
    if output_key not in outputs:
        return None, None
    
    output = outputs[output_key]
    
    # Splice outputs have different structure: {'probs': tensor} or {'predictions': tensor}
    if output_type == OutputType.SPLICE_SITES:
        # splice_sites_classification returns {'logits': ..., 'probs': ...}
        # shape is (B, S, 5)
        return output['probs'], 1
    elif output_type == OutputType.SPLICE_SITE_USAGE:
        # splice_sites_usage returns {'logits': ..., 'predictions': ...}
        return output['predictions'], 1
    
    # Standard outputs: {1: tensor, 128: tensor} or {128: tensor}
    if isinstance(output, dict):
        # Use preferred_resolution if specified and available
        if preferred_resolution is not None and preferred_resolution in output:
            return output[preferred_resolution], preferred_resolution
        
        # Otherwise use default preference (128bp for compatibility)
        if 128 in output:
            return output[128], 128
        elif 1 in output:
            return output[1], 1
    
    return output, 128  # Fallback



def get_splice_site_metadata():
    """Generate synthetic metadata for splice site classification channels."""
    # Channel order from heads.py: Donor+, Acceptor+, Donor-, Acceptor-, Other
    # indices: 0, 1, 2, 3, 4
    tracks = [
        ('donor', '+', 0),
        ('acceptor', '+', 1),
        ('donor', '-', 2),
        ('acceptor', '-', 3),
        # ('other', '.', 4) # We skip 'other' for visualization
    ]
    
    metadata_list = []
    for name, strand, idx in tracks:
        # Create a simplified object that mimics TrackMetadata
        m = type('TrackMetadata', (), {})()
        m.track_name = name
        m.track_strand = strand
        m.track_index = idx
        m.biosample_name = ''
        m.ontology_curie = ''
        m.assay_title = 'Splice Site Classification'
        m.transcription_factor = ''
        m.histone_mark = ''
        metadata_list.append(m)
    return metadata_list


def visualize_variant(
    scoring_model,
    interval,
    variant,
    # Ontology filtering
    ontology_terms=None,
    # Gene annotation options
    plot_gene_annotation=True,
    plot_longest_transcript_only=True,
    # Output types to plot
    plot_cage=True,
    plot_rna_seq=True,
    plot_splice_sites=True,
    plot_atac=False,
    plot_dnase=False,
    plot_chip_histone=False,
    plot_chip_tf=False,
    plot_splice_site_usage=False,
    # Strand filtering
    filter_to_positive_strand=False,
    filter_to_negative_strand=True,
    # Resolution
    resolution=None,  # None = auto (prefers 128bp), 1 = 1bp, 128 = 128bp
    # Plot options
    mode='overlay',  # 'overlay' (REF vs ALT) or 'diff' (ALT - REF)
    ref_color='dimgrey',
    alt_color='red',
    diff_color='dimgrey',
    filled=True,
    plot_interval_width=43008,
    plot_interval_shift=0,
    organism='human',
):
    """Visualize variant effects on genomic tracks.

    Args:
        scoring_model: VariantScoringModel instance
        interval: Interval to score
        variant: Variant to score
        ontology_terms: List of ontology CURIEs to filter tracks (e.g., ['EFO:0001187'])
        plot_gene_annotation: Whether to plot gene annotation track
        plot_longest_transcript_only: Use only longest transcript per gene
        plot_cage: Plot CAGE tracks
        plot_rna_seq: Plot RNA-seq tracks
        plot_splice_sites: Plot splice site classification
        plot_atac: Plot ATAC-seq tracks
        plot_dnase: Plot DNase-seq tracks
        plot_chip_histone: Plot ChIP-seq histone tracks
        plot_chip_tf: Plot ChIP-seq TF tracks
        plot_splice_site_usage: Plot splice site usage
        filter_to_positive_strand: Show only + strand tracks
        filter_to_negative_strand: Show only - strand tracks
        resolution: Output resolution (1 or 128). If None, prefers 128bp when available.
        mode: 'overlay' shows REF and ALT overlaid, 'diff' shows ALT - REF difference.
        ref_color: Color for REF allele (overlay mode)
        alt_color: Color for ALT allele (overlay mode)
        diff_color: Color for difference tracks (diff mode)
        filled: Whether to fill the area under tracks (diff mode). Default True.
        plot_interval_width: Width of plot window (bp)
        plot_interval_shift: Shift plot window center by this amount (bp)
        organism: 'human' or 'mouse'
    """
    
    if filter_to_positive_strand and filter_to_negative_strand:
        raise ValueError('Cannot filter to both positive and negative strand.')
    
    # Get raw predictions
    ref_outputs, alt_outputs = scoring_model.predict_variant(interval, variant, to_cpu=True)
    
    # Get all metadata
    all_metadata = scoring_model.get_track_metadata()
    
    # Create JAX interval and variant
    jax_interval = jax_genome.Interval(interval.chromosome, interval.start, interval.end)
    jax_variant = jax_genome.Variant(
        chromosome=variant.chromosome,
        position=variant.position,
        reference_bases=variant.reference_bases,
        alternate_bases=variant.alternate_bases,
    )
    
    ref_alt_colors = {'REF': ref_color, 'ALT': alt_color}
    components = []
    
    # Gene annotation (uses scoring_model's annotation if available)
    if plot_gene_annotation:
        _, longest_extractor = get_transcript_extractors(scoring_model, organism)
        transcripts = longest_extractor.extract(jax_interval)
        components.append(plot_components.TranscriptAnnotation(transcripts))
    
    # Helper to create filtered TrackData
    def get_track_data(output_type, label):
        ref_preds, res = extract_predictions(ref_outputs, output_type, preferred_resolution=resolution)
        alt_preds, _ = extract_predictions(alt_outputs, output_type, preferred_resolution=resolution)
        
        if ref_preds is None or alt_preds is None:
            return None, None
        
        if output_type == OutputType.SPLICE_SITES:
            # SPLICE_SITES won't be in loaded metadata, synthesize it
            metadata_list = get_splice_site_metadata()
        else:
            metadata_list = all_metadata.get(output_type, [])
        
        if not metadata_list:
            return None, None
        
        # Filter by ontology (only for standard tracks)
        if ontology_terms and output_type != OutputType.SPLICE_SITES:
            indices = [i for i, m in enumerate(metadata_list) if m.ontology_curie in ontology_terms]
            if not indices:
                return None, None
            metadata_list = [metadata_list[i] for i in indices]
            ref_preds = ref_preds[:, :, indices]
            alt_preds = alt_preds[:, :, indices]
        
        # Filter by strand
        if filter_to_positive_strand or filter_to_negative_strand:
            # Find indices where strand matches
            indices = []
            filtered_meta = []

            for i, m in enumerate(metadata_list):
                if filter_to_positive_strand:
                    keep = m.track_strand == '+'
                else:
                    # Negative strand filter: keep '-' and '.' (unstranded)
                    keep = m.track_strand != '+'
                if keep:
                    indices.append(i)
                    filtered_meta.append(m)
            
            if not indices:
                return None, None
                
            metadata_list = filtered_meta
            
            # Slice predictions based on indices
            # Handle special case for SPLICE_SITES which uses indices as channel map
            if output_type == OutputType.SPLICE_SITES:
                # metadata_list[i].track_index holds the channel index
                channel_indices = [m.track_index for m in metadata_list]
                ref_preds = ref_preds[:, :, channel_indices]
                alt_preds = alt_preds[:, :, channel_indices]
            else:
                # Normal indexing
                ref_preds = ref_preds[:, :, indices]
                alt_preds = alt_preds[:, :, indices]
        
        ref_tdata = pytorch_to_track_data(ref_preds, metadata_list, interval, res, label)
        alt_tdata = pytorch_to_track_data(alt_preds, metadata_list, interval, res, label)

        if mode == 'diff':
            # Compute ALT - REF difference
            diff_values = alt_tdata.values - ref_tdata.values
            diff_tdata = jax_track_data.TrackData(
                values=diff_values,
                metadata=ref_tdata.metadata,
                resolution=ref_tdata.resolution,
                interval=ref_tdata.interval,
            )
            return diff_tdata, None

        return ref_tdata, alt_tdata
    
    # Plot map
    plot_map = {
        'plot_cage': (OutputType.CAGE, 'CAGE', '{output_type}: {biosample_name} ({strand})\\n{name}'),
        'plot_rna_seq': (OutputType.RNA_SEQ, 'RNA_SEQ', '{output_type}: {biosample_name} ({strand})\\n{name}'),
        'plot_atac': (OutputType.ATAC, 'ATAC', '{output_type}: {biosample_name} ({strand})\\n{name}'),
        'plot_dnase': (OutputType.DNASE, 'DNASE', '{output_type}: {biosample_name} ({strand})\\n{name}'),
        'plot_chip_histone': (OutputType.CHIP_HISTONE, 'CHIP_HISTONE', '{output_type}: {biosample_name} ({strand})\\n{histone_mark}'),
        'plot_chip_tf': (OutputType.CHIP_TF, 'CHIP_TF', '{output_type}: {biosample_name} ({strand})\\n{transcription_factor}'),
        'plot_splice_sites': (OutputType.SPLICE_SITES, 'SPLICE_SITES', '{output_type}: {name} ({strand})'),
        'plot_splice_site_usage': (OutputType.SPLICE_SITE_USAGE, 'SPLICE_SITE_USAGE', 'SPLICE_SITE_USAGE: {name} ({strand})'),
    }
    
    for plot_flag, (output_type, label, ylabel_template) in plot_map.items():
        if not locals().get(plot_flag, False): # Use locals() to access args
            continue

        data_a, data_b = get_track_data(output_type, label)
        if data_a is None:
            continue

        if data_a.values.shape[-1] == 0:
            continue

        if mode == 'diff':
            component = plot_components.Tracks(
                tdata=data_a,  # Already contains ALT - REF diff
                filled=filled,
                track_colors=diff_color,
                ylabel_template=ylabel_template,
            )
        else:
            component = plot_components.OverlaidTracks(
                tdata={'REF': data_a, 'ALT': data_b},
                colors=ref_alt_colors,
                ylabel_template=ylabel_template,
            )
        components.append(component)
    
    
    if plot_interval_width > interval.width:
        raise ValueError(
            f'plot_interval_width ({plot_interval_width}) must be less than '
            f'interval.width ({interval.width}).'
        )
    
    # Plot
    plot_interval = jax_interval.shift(plot_interval_shift).resize(plot_interval_width)
    
    return plot_components.plot(
        components=components,
        interval=plot_interval,
        annotations=[plot_components.VariantAnnotation([jax_variant])],
    )

# Visualization imports and helpers
from alphagenome.data import track_data as jax_track_data
from alphagenome.data import genome as jax_genome
from alphagenome.data import gene_annotation, transcript
from alphagenome.visualization import plot_components
from .types import OutputType
import pandas as pd
import numpy as np
import torch
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logomaker

# Cache for transcript extractors
_transcript_extractor_cache = {}
BASE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

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


def extract_ism_logo(
    ism_results,
    scorer_idx,
    track_idx,
    background_seq,
    center_rel,
    half_window,
    interval,
    is_gene_scorer=False,
    gene_id=None
):
    """Convert ISM results to a logomaker-compatible (N, 4) DataFrame.
    Alt columns receive the raw ISM score; the ref-base column is set to 0.
    Then each row is mean-centred by subtracting the mean of the three alt
    scores from all four entries (see AlphaGenome Supplementary Methods).

    Args:
        ism_results: List of per-variant score lists from an ISM run (e.g. the
            output of ``scoring_model.score_ism_variants`` or an equivalent
            manual ISM loop).  Each element is a list of score objects, one per
            scorer.
        scorer_idx: Index into each variant's score list selecting which scorer
            to read (e.g. 0 for DNase, 1 for ChIP-histone, 2 for RNA-seq when
            using the standard ``ism_scorers`` list).
        track_idx: Index of the specific output track within the chosen scorer's
            ``scores`` tensor (e.g. the CMP DNase track index).
        background_seq: The full nucleotide sequence string (REF or ALT) that
            was used as the ISM background, spanning the scored ``interval``.
        center_rel: Relative position of the variant center inside the interval,
            i.e. ``variant.start - interval.start``.
        half_window: Half-width of the ISM window in base pairs.  The logo will
            cover ``2 * half_window + 1`` positions centred on ``center_rel``.
        interval: The genomic ``Interval`` over which ISM was performed.  Used
            to convert absolute genomic coordinates in score objects to relative
            positions via ``variant.start - interval.start``.
        is_gene_scorer: If True, the score object at ``scorer_idx`` is expected
            to be a list of per-gene scores (as returned by
            ``GeneMaskLFCScorer``).  The function will search this list for a
            matching ``gene_id``.
        gene_id: Ensembl gene ID without version suffix (e.g.
            ``'ENSG00000162367'``) used to select the correct gene score when
            ``is_gene_scorer`` is True.

    Returns:
        pd.DataFrame: Shape ``(2 * half_window + 1, 4)`` with columns
        ``['A', 'C', 'G', 'T']``, mean-centred per row.
    """
    n_pos = 2 * half_window + 1
    scores = np.zeros((n_pos, 4))
    window_start = center_rel - half_window

    for variant_scores in ism_results:
        score_result = variant_scores[scorer_idx]
        if is_gene_scorer:
            if not isinstance(score_result, list):
                score_result = [score_result]
            score_obj = None
            for gs in score_result:
                if gs.gene_id and gs.gene_id.split('.')[0] == gene_id:
                    score_obj = gs
                    break
            if score_obj is None:
                continue
        else:
            score_obj = score_result

        v = score_obj.variant
        rel_pos = v.start - interval.start
        win_idx = rel_pos - window_start
        if not (0 <= win_idx < n_pos):
            continue

        alt_idx = BASE_TO_IDX.get(v.alternate_bases.upper())
        if alt_idx is not None:
            scores[win_idx, alt_idx] = score_obj.scores[track_idx].item()

    # Mean center each row: subtract mean of the 3 alt scores
    for i in range(n_pos):
        rp = window_start + i
        if 0 <= rp < len(background_seq):
            ref_base = background_seq[rp].upper()
            ref_idx = BASE_TO_IDX.get(ref_base)
            if ref_idx is not None:
                alt_indices = [j for j in range(4) if j != ref_idx]
                mean_alts = np.mean([scores[i, j] for j in alt_indices])
                for j in range(4):
                    scores[i, j] -= mean_alts

    return pd.DataFrame(scores, columns=list('ACGT'))


def build_ism_logos(
    ism_ref,
    ism_alt,
    ref_seq,
    alt_seq,
    interval,
    track_configs,
    center_rel,
    half_window,
    gene_id,
):
    """Build and normalize REF and ALT ISM logos.

    Args:
    Args:
        ism_ref: List of per-variant score lists from ISM on the reference
            background sequence, as returned by
            ``scoring_model.score_ism_variants`` or an equivalent loop.
        ism_alt: List of per-variant score lists from ISM on the ALT background
            sequence (variant applied).
        ref_seq: Full reference nucleotide sequence string spanning the interval.
        alt_seq: Full ALT nucleotide sequence string (with the variant applied)
            spanning the interval, truncated to ``interval.width``.
        interval: The genomic ``Interval`` over which ISM was performed.  Used
            to convert absolute genomic coordinates in score objects to relative
            positions via ``variant.start - interval.start``.
        track_configs: List of ``(scorer_idx, track_idx, label)`` tuples
            identifying which scorer/track combinations to build logos for (e.g.
            ``[(0, 44, 'DNase'), (1, 206, 'H3K27ac'), (2, 561, 'TAL1 RNA-seq')]``).
            Whether a scorer is a gene scorer is auto-detected from the score
            object type (list vs scalar).
        center_rel: Relative position of the variant center inside the interval,
            i.e. ``variant.start - interval.start``.
        half_window: Half-width of the ISM window in base pairs.  The logo will
            cover ``2 * half_window + 1`` positions centred on ``center_rel``.
        gene_id: Ensembl gene ID without version suffix (e.g.
            ``'ENSG00000162367'``), passed to ``extract_ism_logo`` for gene
            scorer lookup.

    Returns:
        tuple[dict, dict]: ``(ref_logos, alt_logos)`` where each dict is keyed
        by track label and maps to a ``pd.DataFrame`` of shape
        ``(2 * half_window + 1, 4)``.  Values are normalized so the peak
        absolute value across both REF and ALT logos equals 1.
    """
    ref_logos, alt_logos = {}, {}
    for scorer_idx, track_idx, label in track_configs:
        is_gene = isinstance(ism_ref[0][scorer_idx], list)
        ref_logos[label] = extract_ism_logo(
            ism_ref,
            scorer_idx,
            track_idx,
            ref_seq,
            center_rel,
            half_window,
            interval,
            is_gene_scorer=is_gene,
            gene_id=gene_id,
        )
        alt_logos[label] = extract_ism_logo(
            ism_alt,
            scorer_idx,
            track_idx,
            alt_seq,
            center_rel,
            half_window,
            interval,
            is_gene_scorer=is_gene,
            gene_id=gene_id,
        )

    # Normalize: divide by max |value| across REF+ALT so peak = 1
    for _, _, label in track_configs:
        combined = pd.concat([ref_logos[label], alt_logos[label]])
        max_abs = np.abs(combined.values).max()
        if max_abs > 0:
            ref_logos[label] = ref_logos[label] / max_abs
            alt_logos[label] = alt_logos[label] / max_abs

    return ref_logos, alt_logos


def plot_ism_logos(
    ref_logos,
    alt_logos,
    track_configs,
    variant,
    half_window,
    title='ISM sequence logos — REF vs ALT background',
    display_names=None,
    savepath=None,
):
    """Plot ISM sequence logos.

    Renders a two-panel figure (REF on top, ALT on bottom) with one column of
    logomaker sequence logos per track, separated by a variant annotation row.

    Args:
        ref_logos: Dict mapping track label → ``pd.DataFrame`` of shape
            ``(2 * half_window + 1, 4)`` with columns ``['A', 'C', 'G', 'T']``,
            as returned by ``build_logos``.
        alt_logos: Same structure as ``ref_logos`` but computed on the ALT
            background sequence.
        track_configs: List of ``(scorer_idx, track_idx, label)`` tuples.  Only
            the ``label`` element is used for axis labelling and dict lookup.
        variant: A ``Variant`` object (or equivalent) with ``.position``,
            ``.reference_bases``, and ``.alternate_bases`` attributes, used for
            the annotation marker and genomic-coordinate x-axis labels.
        half_window: Half-width of the ISM window in base pairs.  Determines
            the x-axis range: ``variant.position ± half_window``.
        title: Figure suptitle string.
        display_names: Optional dict mapping track config labels to shorter
            y-axis display names.
        savepath: If not None, the figure is saved to this path at 150 dpi.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    track_labels   = [label for _, _, label in track_configs]

    ylims = {}
    for label in track_labels:
        all_vals = pd.concat([ref_logos[label], alt_logos[label]])
        vmin, vmax = all_vals.values.min(), all_vals.values.max()
        pad = (vmax - vmin) * 0.05
        ylims[label] = (vmin - pad, vmax + pad)

    window_positions = list(range(
        variant.position - half_window,
        variant.position + half_window + 1,
    ))
    alt_len = len(variant.alternate_bases)

    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(
        7, 1,
        height_ratios=[1, 1, 1, 0.3, 1, 1, 1],
        hspace=0.05,
    )
    row_map = {
        (0, 0): 0, (0, 1): 1, (0, 2): 2,
        (1, 0): 4, (1, 1): 5, (1, 2): 6,
    }
    first_ax = None
    axes = {}
    for key, row in row_map.items():
        if first_ax is None:
            axes[key] = fig.add_subplot(gs[row])
            first_ax = axes[key]
        else:
            axes[key] = fig.add_subplot(gs[row], sharex=first_ax)

    ax_gap = fig.add_subplot(gs[3])
    ax_gap.set_xlim(-0.5, 2 * half_window + 0.5)
    ax_gap.set_ylim(0, 1)
    ax_gap.axis('off')
    ax_gap.plot(half_window + (alt_len - 1) / 2, 0.9, marker='v',
                color='darkorange', markersize=12, zorder=5, clip_on=False)
    ax_gap.text(
        half_window + alt_len + 1, 0.45,
        f'Chr. 1: {variant.position}: {variant.reference_bases} > {variant.alternate_bases}',
        fontsize=9, color='darkorange', va='center',
    )

    for group_idx, (logo_dict, bg_label) in enumerate(
        [(ref_logos, 'REF'), (alt_logos, 'ALT')]
    ):
        for tidx, label in enumerate(track_labels):
            ax = axes[(group_idx, tidx)]
            df = logo_dict[label].copy()
            ylo, yhi = ylims[label]
            df = df.clip(lower=ylo, upper=yhi)
            df.index = range(len(df))

            logomaker.Logo(
                df, ax=ax, shade_below=0.5, fade_below=0.5,
                color_scheme='classic',
            )
            ax.axvspan(
                half_window - 0.5, half_window + alt_len - 0.5,
                color='#f5deb3', alpha=0.5, zorder=0,
            )
            ax.set_xlim(-0.5, len(df) - 0.5)
            ax.set_ylim(ylo - 0.25, yhi + 0.25)
            ax.axhline(0, color='grey', linewidth=0.3)
            ax.set_ylabel((display_names or {}).get(label, label), fontsize=9, rotation=0,
                          labelpad=55, va='center')

            if group_idx == 1 and tidx == len(track_labels) - 1:
                tick_step = 10
                tick_idx = list(range(0, len(df), tick_step))
                ax.set_xticks(tick_idx)
                ax.set_xticklabels(
                    [f'{window_positions[i]:,}' for i in tick_idx],
                    rotation=45, ha='right', fontsize=7,
                )
                ax.set_xlabel('Genomic position (chr1)', fontsize=9)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    for group_idx, bg_label in enumerate(['REF', 'ALT']):
        ax = axes[(group_idx, 1)]
        ax.annotate(
            bg_label, xy=(0, 0.5), xycoords='axes fraction',
            xytext=(-100, 0), textcoords='offset points',
            fontsize=13, rotation=90,
            va='center', ha='center',
        )

    fig.suptitle(title, fontsize=14)
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.show()
    return fig

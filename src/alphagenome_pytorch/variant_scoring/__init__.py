"""Variant scoring for AlphaGenome PyTorch model.

This module provides functionality for scoring the effects of genetic variants
using the AlphaGenome model. It supports various scoring strategies including
spatial masking, gene-level scoring, and contact map analysis.

Example:
    >>> from alphagenome_pytorch import AlphaGenome
    >>> from alphagenome_pytorch.variant_scoring import (
    ...     VariantScoringModel, Variant, Interval,
    ...     CenterMaskScorer, ContactMapScorer,
    ...     OutputType, AggregationType,
    ...     get_recommended_scorers,
    ... )
    >>>
    >>> # Load model (track_means are bundled in weights from convert_weights.py)
    >>> model = AlphaGenome()
    >>> model.load_state_dict(torch.load('model.pth'))
    >>>
    >>> # Create scoring wrapper with FASTA and GTF
    >>> scoring_model = VariantScoringModel(
    ...     model,
    ...     fasta_path='hg38.fa',
    ...     gtf_path='gencode.gtf',
    ... )
    >>>
    >>> # Define variant and interval
    >>> variant = Variant.from_str('chr22:36201698:A>C')
    >>> interval = Interval.centered_on('chr22', 36201698, width=131072)
    >>>
    >>> # Score with recommended scorers
    >>> scores = scoring_model.score_variant(
    ...     interval, variant,
    ...     scorers=get_recommended_scorers('human'),
    ...     organism_index=0,
    ... )
"""

# Core types
from .types import (
    AggregationType,
    Interval,
    OutputType,
    TrackMetadata,
    Variant,
    VariantScore,
    load_track_metadata,
    scores_to_anndata,
    scores_to_dataframe,
    tidy_scores,
    Width,
)

# Aggregation functions
from .aggregations import align_alternate, compute_aggregation, create_center_mask

# Sequence utilities
from .sequence import (
    FastaExtractor,
    apply_variant_to_onehot,
    apply_variant_to_sequence,
    extract_sequence_from_fasta,
    onehot_to_sequence,
    sequence_to_onehot,
)

# Gene annotations
from .annotations import GeneAnnotation, GeneInfo, PolyAAnnotation

# Scorers
from .scorers import (
    BaseVariantScorer,
    CenterMaskScorer,
    ContactMapScorer,
    GeneMaskActiveScorer,
    GeneMaskLFCScorer,
    GeneMaskSplicingScorer,
    PolyadenylationScorer,
    SpliceJunctionScorer,
)
from .scorers.gene_mask import GeneMaskMode

# Inference wrapper
from .inference import (
    RECOMMENDED_VARIANT_SCORERS,
    VariantScoringModel,
    get_recommended_scorers,
)

# Visualization utilities are loaded lazily. ``visualization_utils`` imports
# matplotlib at module scope (heavy, and triggers a one-off font-cache build on
# first use), yet most code that transitively imports ``variant_scoring`` — e.g.
# the serving/adapter CLI paths — never plots. PEP 562 keeps
# ``from alphagenome_pytorch.variant_scoring import visualize_variant`` working
# while deferring the matplotlib import until the name is actually accessed.
def __getattr__(name: str):
    if name == "visualize_variant":
        from .visualization_utils import visualize_variant

        return visualize_variant
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Types
    'AggregationType',
    'Interval',
    'OutputType',
    'TrackMetadata',
    'Variant',
    'VariantScore',
    'load_track_metadata',
    'scores_to_anndata',
    'scores_to_dataframe',
    'tidy_scores',
    'Width',
    # Aggregations
    'align_alternate',
    'compute_aggregation',
    'create_center_mask',
    # Sequence
    'FastaExtractor',
    'apply_variant_to_onehot',
    'apply_variant_to_sequence',
    'extract_sequence_from_fasta',
    'onehot_to_sequence',
    'sequence_to_onehot',
    # Annotations
    'GeneAnnotation',
    'GeneInfo',
    'PolyAAnnotation',
    # Scorers
    'BaseVariantScorer',
    'CenterMaskScorer',
    'ContactMapScorer',
    'GeneMaskActiveScorer',
    'GeneMaskLFCScorer',
    'GeneMaskMode',
    'GeneMaskSplicingScorer',
    'PolyadenylationScorer',
    'SpliceJunctionScorer',
    # Inference
    'RECOMMENDED_VARIANT_SCORERS',
    'VariantScoringModel',
    'get_recommended_scorers',
    # Visualization
    'visualize_variant',
]

"""Variant scoring inference wrapper for AlphaGenome model.

This module provides the main entry point for scoring variants using
the AlphaGenome PyTorch model.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .annotations import GeneAnnotation
from .scorers.base import BaseVariantScorer
from .sequence import FastaExtractor, apply_variant_to_sequence, sequence_to_onehot
from .types import Interval, OutputType, TrackMetadata, Variant, VariantScore, tidy_scores

if TYPE_CHECKING:
    from ..model import AlphaGenome


class VariantScoringModel:
    """Wrapper for AlphaGenome model with variant scoring utilities.

    This class provides a convenient interface for:
    - Extracting reference and alternate sequences
    - Running model inference
    - Computing variant scores using various scorers

    Example:
        First, generate the required metadata files:

            # Convert GTF to Parquet for gene annotations
            $ python scripts/convert_gtf_to_parquet.py \\
                --input gencode.v49.gtf \\
                --output gencode.v49.parquet

            # Extract track metadata from the model (bfloat16/float32)
            $ python scripts/extract_track_metadata.py \\
                --output-file track_metadata.parquet

        Then run scoring:

        >>> from alphagenome_pytorch import AlphaGenome
        >>> from alphagenome_pytorch.variant_scoring import (
        ...     VariantScoringModel, Variant, Interval,
        ...     CenterMaskScorer, OutputType, AggregationType,
        ... )
        >>>
        >>> # Load model (track_means are bundled in weights from convert_weights.py)
        >>> model = AlphaGenome()
        >>> model.load_state_dict(torch.load('model.pth'))
        >>>
        >>> # Create scoring wrapper (defaults to Human)
        >>> scorer = VariantScoringModel(
        ...     model,
        ...     fasta_path='hg38.fa',
        ...     gtf_path='gencode.v49.parquet'
        ... )
        >>> scorer.load_all_metadata('track_metadata.parquet')
        >>>
        >>> # Score a variant
        >>> variant = Variant('chr22', 36201698, 'A', 'C')
        >>> interval = Interval('chr22', 36136162, 36267234)
        >>> scores = scorer.score_variant(
        ...     interval, variant,
        ...     scorers=[
        ...         CenterMaskScorer(OutputType.ATAC, 501, AggregationType.DIFF_LOG2_SUM)
        ...     ],
        ... )
    """

    def __init__(
        self,
        model: 'AlphaGenome',
        fasta_path: str | Path | None = None,
        gtf_path: str | Path | None = None,
        polya_path: str | Path | None = None,
        device: str | torch.device | None = None,
        default_organism: str | int | None = 'human',
    ):
        """Initialize variant scoring wrapper.

        Args:
            model: AlphaGenome model instance
            fasta_path: Path to reference genome FASTA file. Required for
                sequence extraction. If None, sequences must be provided directly.
            gtf_path: Path to GTF annotation file for gene-centric scorers.
                If None, gene-centric scorers will require passing gene_annotation.
            polya_path: Path to GENCODE polyAs GTF/Parquet file for PolyadenylationScorer.
                If None, PolyadenylationScorer will use peak detection fallback.
            device: Device to run inference on. If None, uses model's device.
            default_organism: Default organism to use for scoring ('human', 'mouse', or index).
        """
        self.model = model
        
        # Determine supported organisms from model
        self.num_organisms = getattr(model, 'num_organisms', 2)
        
        # Map common names to indices
        self.organism_map = {
            'human': 0,
            'homo_sapiens': 0,
            'mouse': 1,
            'mus_musculus': 1,
        }
        
        # Validate/Set default organism
        self.default_organism_index = None
        if default_organism is not None:
            self.default_organism_index = self._resolve_organism_index(default_organism)

        if device is None:
            # Get device from model parameters
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
        self.device = torch.device(device)

        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Initialize FASTA extractor if path provided
        self._fasta_extractor: FastaExtractor | None = None
        if fasta_path is not None:
            self._fasta_extractor = FastaExtractor(str(fasta_path))

        # Initialize gene annotation if path provided
        self._gene_annotation: GeneAnnotation | None = None
        if gtf_path is not None:
            self._gene_annotation = GeneAnnotation(gtf_path)

        # Initialize polyA annotation if path provided
        self._polya_annotation: 'PolyAAnnotation | None' = None
        if polya_path is not None:
            from .annotations import PolyAAnnotation
            self._polya_annotation = PolyAAnnotation(polya_path)

        # Track metadata for tidy_scores output
        # Keyed by organism_index (0=Human, 1=Mouse) -> OutputType -> List[TrackMetadata]
        self._track_metadata: dict[int, dict[OutputType, list[TrackMetadata]]] = {0: {}, 1: {}}

    def _resolve_organism_index(self, organism: str | int | None) -> int:
        """Resolve organism identifier to an index."""
        if organism is None:
            if self.default_organism_index is not None:
                return self.default_organism_index
            # Default to 0 (Human) if no default set, for backward compatibility
            return 0
            
        if isinstance(organism, int):
            idx = organism
        elif isinstance(organism, str):
            normalized = organism.lower()
            if normalized in self.organism_map:
                idx = self.organism_map[normalized]
            else:
                try:
                    idx = int(organism)
                except ValueError:
                    raise ValueError(f"Unknown organism name: {organism}")
        else:
             raise ValueError(f"Invalid organism type: {type(organism)}")
             
        if idx < 0 or idx >= self.num_organisms:
            raise ValueError(
                f"Organism index {idx} out of range for model with {self.num_organisms} organisms"
            )
        return idx
        
    @property
    def fasta(self) -> FastaExtractor:
        """FASTA extractor for sequence extraction."""
        if self._fasta_extractor is None:
            raise ValueError(
                "FASTA path not provided. Initialize with fasta_path parameter "
                "or provide sequences directly."
            )
        return self._fasta_extractor

    @property
    def gene_annotation(self) -> GeneAnnotation | None:
        """Gene annotation for gene-centric scorers."""
        return self._gene_annotation

    @property
    def polya_annotation(self) -> 'PolyAAnnotation | None':
        """PolyA annotation for PolyadenylationScorer."""
        return self._polya_annotation

    @property
    def track_metadata(self) -> dict[OutputType, list[TrackMetadata]]:
        """Track metadata for default organism (or Human)."""
        idx = self.default_organism_index if self.default_organism_index is not None else 0
        return self._track_metadata.get(idx, {})
    
    def get_track_metadata(self, organism: str | int | None = None) -> dict[OutputType, list[TrackMetadata]]:
        """Get track metadata for a specific organism."""
        idx = self._resolve_organism_index(organism)
        return self._track_metadata.get(idx, {})

    def set_track_metadata(
        self,
        output_type: OutputType,
        metadata: list[TrackMetadata],
        organism: str | int | None = None,
    ) -> None:
        """Set track metadata for an output type."""
        idx = self._resolve_organism_index(organism)
        if idx not in self._track_metadata:
            self._track_metadata[idx] = {}
        self._track_metadata[idx][output_type] = metadata

    def load_track_metadata(
        self,
        metadata_path: str | Path,
        output_type: OutputType,
        organism: str | int | None = None,
    ) -> None:
        """Load track metadata from a file."""
        from .types import load_track_metadata as _load_track_metadata
        
        idx = self._resolve_organism_index(organism)
        if idx not in self._track_metadata:
            self._track_metadata[idx] = {}
            
        self._track_metadata[idx][output_type] = _load_track_metadata(
            str(metadata_path), output_type
        )
        
    def load_all_metadata(self, metadata_path: str | Path) -> None:
        """Load all track metadata from a single parquet file.
        
        Args:
            metadata_path: Path to the parquet file containing all metadata.
                Must contain 'organism' and 'output_type' columns.
        """
        import pandas as pd
        
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
            
        df = pd.read_parquet(path)
        
        if 'organism' not in df.columns or 'output_type' not in df.columns:
             raise ValueError("Metadata file must contain 'organism' and 'output_type' columns")

        for (org_name, output_type_str), group_df in df.groupby(['organism', 'output_type']):
            try:
                # Resolve organism string to index using our map
                org_idx = self._resolve_organism_index(org_name)
            except ValueError:
                continue # Skip unknown organisms in file
                
            try:
                if output_type_str == 'contact_maps':
                    output_type = OutputType.CONTACT_MAPS
                else:
                    output_type = OutputType(output_type_str)
            except ValueError:
                continue

            if org_idx not in self._track_metadata:
                self._track_metadata[org_idx] = {}
            
            tracks = []
            for i, (_, row) in enumerate(group_df.iterrows()):
                meta = TrackMetadata(
                    track_index=i,
                    track_name=row.get('track_name', f'track_{i}'),
                    track_strand=row.get('track_strand', '.'),
                    output_type=output_type,
                    ontology_curie=row.get('ontology_curie'),
                    gtex_tissue=row.get('gtex_tissue'),
                    assay_title=row.get('assay_title'),
                    biosample_name=row.get('biosample_name'),
                    biosample_type=row.get('biosample_type'),
                    transcription_factor=row.get('transcription_factor'),
                    histone_mark=row.get('histone_mark'),
                )
                tracks.append(meta)
                
            self._track_metadata[org_idx][output_type] = tracks

    def get_sequence(
        self,
        interval: Interval,
        variant: Variant | None = None,
    ) -> str:
        """Get DNA sequence for an interval, optionally with variant applied."""
        seq = self.fasta.extract(interval)

        if variant is not None:
            seq = apply_variant_to_sequence(seq, variant, interval)

        return seq

    @torch.no_grad()
    def predict(
        self,
        sequence: str | torch.Tensor,
        organism: str | int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run model inference on a sequence.

        Args:
            sequence: DNA sequence (string or one-hot tensor)
            organism: 'human', 'mouse', or index. Uses default_organism if None.

        Returns:
            Dictionary of model outputs
        """
        organism_index = self._resolve_organism_index(organism)
        
        if isinstance(sequence, str):
            onehot = sequence_to_onehot(
                sequence,
                dtype=self.model.dtype_policy.compute_dtype,
                device=self.device,
            )
        else:
            onehot = sequence.to(dtype=self.model.dtype_policy.compute_dtype, device=self.device)

        # Add batch dimension if needed
        if onehot.dim() == 2:
            onehot = onehot.unsqueeze(0)

        # Create organism index tensor
        batch_size = onehot.shape[0]
        org_idx = torch.full(
            (batch_size,),
            organism_index,
            dtype=torch.long,
            device=self.device,
        )

        return self.model(onehot, org_idx, **kwargs)

    @torch.no_grad()
    def predict_variant(
        self,
        interval: Interval,
        variant: Variant,
        organism: str | int | None = None,
        to_cpu: bool = False,
        unified_splicing: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Get reference and alternate predictions for a variant.

        Args:
            interval: Genomic interval containing the variant.
            variant: Variant to score.
            organism: 'human', 'mouse', or index. Uses default if None.
            to_cpu: If True, move all output tensors to CPU and clear GPU cache.
                Recommended for visualization or when processing many variants.
            unified_splicing: If True, performs a second pass to align splice
                sites between Ref and Alt predictions. Required for SpliceJunctionScorer.

        Returns:
            Tuple of (ref_outputs, alt_outputs) dictionaries.
        """
        # Handle indels: for deletions, extend extraction to compensate for
        # sequence shrinkage. For insertions, the alt is longer and gets
        # truncated. Both ref and alt are truncated to the original interval
        # length to ensure fixed-length input to the model.
        interval_length = interval.width
        deletion_extension = max(
            0, len(variant.reference_bases) - len(variant.alternate_bases)
        )

        if deletion_extension > 0:
            extraction_interval = Interval(
                interval.chromosome,
                interval.start,
                interval.end + deletion_extension,
            )
        else:
            extraction_interval = interval

        base_seq = self.fasta.extract(extraction_interval)
        ref_seq = base_seq[:interval_length]
        alt_seq = apply_variant_to_sequence(
            base_seq, variant, extraction_interval
        )[:interval_length]

        # First pass: Get standard predictions with embeddings if needed
        return_embeddings = unified_splicing
        ref_outputs = self.predict(ref_seq, organism, return_embeddings=return_embeddings)
        alt_outputs = self.predict(alt_seq, organism, return_embeddings=return_embeddings)

        if unified_splicing:
            # Calculate unified splice site positions (max of Ref and Alt)
            from ..utils.splicing import generate_splice_site_positions
            
            ref_probs = ref_outputs['splice_sites_classification']['probs']
            alt_probs = alt_outputs['splice_sites_classification']['probs']
            
            # Use max(Ref, Alt) to determine sites
            unified_positions = generate_splice_site_positions(
                ref=ref_probs,
                alt=alt_probs,
                true_splice_sites=None,
                k=512,
                pad_to_length=512,
                threshold=0.1,
            )
            
            # Second pass: Re-run just the splice junction head with unified positions
            organism_index = self._resolve_organism_index(organism)
            
            def _run_head_with_positions(outputs, pos):
                # Ensure we have the embeddings
                if 'embeddings_1bp' not in outputs:
                     raise ValueError("Embeddings missing from first pass")

                emb = outputs['embeddings_1bp']
                batch_size = emb.shape[0]
                org_idx = torch.full((batch_size,), organism_index, dtype=torch.long, device=self.device)

                # Run just the junction head
                # Note: The head expects embeddings_1bp in NCL format (B, C, S)
                # but predict() returns them in NLC format (B, S, C) when channels_last=True
                # Transpose to NCL format for the head
                emb_ncl = emb.transpose(1, 2)
                junction_out = self.model.splice_sites_junction_head(
                    emb_ncl,
                    org_idx,
                    splice_site_positions=pos
                )
                
                # Update outputs
                outputs['splice_sites_junction'] = junction_out
                return outputs

            ref_outputs = _run_head_with_positions(ref_outputs, unified_positions)
            alt_outputs = _run_head_with_positions(alt_outputs, unified_positions)
            
            # Remove embeddings to save memory / cleanup
            for out in [ref_outputs, alt_outputs]:
                out.pop('embeddings_1bp', None)
                out.pop('embeddings_128bp', None)

        # Move ref to CPU before alt prediction to free GPU memory
        # logic for to_cpu handled at end or between
        
        if to_cpu:
            ref_outputs = self._outputs_to_cpu(ref_outputs)
            alt_outputs = self._outputs_to_cpu(alt_outputs)
            gc.collect()
            torch.cuda.empty_cache()

        return ref_outputs, alt_outputs

    def _outputs_to_cpu(self, outputs: Any) -> Any:
        """Recursively move all tensors in outputs to CPU."""
        if torch.is_tensor(outputs):
            return outputs.cpu()
        if isinstance(outputs, dict):
            return {k: self._outputs_to_cpu(v) for k, v in outputs.items()}
        if isinstance(outputs, list):
            return [self._outputs_to_cpu(v) for v in outputs]
        if isinstance(outputs, tuple):
            return tuple(self._outputs_to_cpu(v) for v in outputs)
        return outputs

    def score_variant(
        self,
        interval: Interval,
        variant: Variant,
        scorers: list[BaseVariantScorer],
        organism: str | int | None = None,
        gene_annotation: GeneAnnotation | None = None,
        to_cpu: bool = False,
    ) -> list[VariantScore | list[VariantScore]]:
        """Score a single variant with multiple scorers.

        Args:
            interval: Genomic interval containing the variant.
            variant: Variant to score
            scorers: List of scorer configurations
            organism: 'human', 'mouse', or index. Uses default_organism if None.
            gene_annotation: Optional GeneAnnotation.
            to_cpu: If True, move scores to CPU and clear GPU cache.

        Returns:
            List of VariantScore objects
        """
        organism_index = self._resolve_organism_index(organism)

        # Check if we need unified splicing pass
        unified_splicing = any(s.name == "SpliceJunctionScorer()" for s in scorers)
        # Note: checking by class name string is fragile but avoids circular imports
        # Alternative: isinstance(s, SpliceJunctionScorer) if imported
        # Let's check typical string representation

        # Get predictions
        ref_outputs, alt_outputs = self.predict_variant(
            interval, variant, organism, unified_splicing=unified_splicing
        )

        # Use instance gene annotation if not provided
        if gene_annotation is None:
            gene_annotation = self._gene_annotation

        # Score with each scorer
        scores = []
        for scorer in scorers:
            score_result = scorer.score(
                ref_outputs=ref_outputs,
                alt_outputs=alt_outputs,
                variant=variant,
                interval=interval,
                organism_index=organism_index,
                gene_annotation=gene_annotation,
                polya_annotation=self._polya_annotation,
            )

            if to_cpu:
                # Move scores to CPU to free GPU memory
                # Handle both single VariantScore and list[VariantScore]
                if isinstance(score_result, list):
                    for s in score_result:
                        if hasattr(s, 'scores') and torch.is_tensor(s.scores):
                            s.scores = s.scores.to(dtype=torch.float32, device='cpu')
                else:
                    s = score_result
                    if hasattr(s, 'scores') and torch.is_tensor(s.scores):
                        s.scores = s.scores.to(dtype=torch.float32, device='cpu')

            scores.append(score_result)

        if to_cpu:
            del ref_outputs, alt_outputs
            torch.cuda.empty_cache()

        return scores

    def score_variants(
        self,
        intervals: list[Interval] | Interval,
        variants: list[Variant],
        scorers: list[BaseVariantScorer],
        organism: str | int | None = None,
        gene_annotation: GeneAnnotation | None = None,
        to_cpu: bool = False,
        progress: bool = True,
    ) -> list[list[VariantScore | list[VariantScore]]]:
        """Score multiple variants with multiple scorers.

        Args:
            intervals: List of intervals or single interval to use for all variants
            variants: List of variants to score
            scorers: List of scorer configurations
            organism: 'human', 'mouse', or index. Uses default_organism if None.
            gene_annotation: Optional GeneAnnotation for gene-centric scorers
            to_cpu: If True, move scores to CPU and clear GPU cache after each variant.
            progress: Whether to show progress bar

        Returns:
            Nested list: outer list is per variant, inner list is per scorer
        """
        # Handle single interval for all variants
        if isinstance(intervals, Interval):
            intervals = [intervals] * len(variants)

        if len(intervals) != len(variants):
            raise ValueError(
                f"Number of intervals ({len(intervals)}) must match "
                f"number of variants ({len(variants)})"
            )

        # Use instance gene annotation if not provided
        if gene_annotation is None:
            gene_annotation = self._gene_annotation

        # Set up progress bar
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    zip(intervals, variants),
                    total=len(variants),
                    desc="Scoring variants",
                )
            except ImportError:
                iterator = zip(intervals, variants)
        else:
            iterator = zip(intervals, variants)

        results = []
        for interval, variant in iterator:
            scores = self.score_variant(
                interval=interval,
                variant=variant,
                scorers=scorers,
                organism=organism,
                gene_annotation=gene_annotation,
                to_cpu=to_cpu,
            )
            results.append(scores)

        return results

    def score_ism_variants(
        self,
        interval: Interval,
        center_position: int,
        scorers: list[BaseVariantScorer],
        window_size: int = 21,
        organism: str | int | None = None,
        gene_annotation: GeneAnnotation | None = None,
        nucleotides: str = 'ACGT',
        to_cpu: bool = True,
        progress: bool = True,
    ) -> list[list[VariantScore | list[VariantScore]]]:
        """Score all possible single-nucleotide mutations in a window.

        In-silico mutagenesis (ISM) systematically evaluates all possible
        SNVs within a window centered on a position of interest. This is
        useful for identifying which positions are most sensitive to mutation.

        Args:
            interval: Genomic interval for prediction context (must be 131072bp)
            center_position: 1-based center position for the ISM window
            scorers: List of scorer configurations
            window_size: Size of the window to mutate (default 21bp, centered)
            organism: 'human', 'mouse', or index. Uses default_organism if None.
            gene_annotation: Optional GeneAnnotation for gene-centric scorers
            nucleotides: Nucleotides to mutate to (default 'ACGT' = all 4 bases)
            to_cpu: If True, move scores to CPU and clear GPU cache after each variant.
            progress: Whether to show progress bar

        Returns:
            Nested list: [variant_idx][scorer_idx] of VariantScore.
            Each variant is a possible SNV in the window.

        Example:
            >>> # Score all SNVs in a 21bp window
            >>> ism_scores = scoring_model.score_ism_variants(
            ...     interval=interval,
            ...     center_position=36201698,
            ...     scorers=[CenterMaskScorer(OutputType.ATAC, 501, AggregationType.DIFF_LOG2_SUM)],
            ...     window_size=21,
            ... )
        """
        # Get reference sequence
        ref_seq = self.get_sequence(interval)

        # Generate all SNVs in window
        variants = []
        half_window = window_size // 2

        for offset in range(-half_window, half_window + 1):
            pos = center_position + offset
            rel_pos = pos - 1 - interval.start  # 0-based relative position

            if 0 <= rel_pos < len(ref_seq):
                ref_base = ref_seq[rel_pos].upper()
                for alt_base in nucleotides:
                    if alt_base.upper() != ref_base:
                        variants.append(Variant(
                            chromosome=interval.chromosome,
                            position=pos,
                            reference_bases=ref_base,
                            alternate_bases=alt_base.upper(),
                        ))

        # Score all variants
        return self.score_variants(
            intervals=interval,
            variants=variants,
            scorers=scorers,
            organism=organism,
            gene_annotation=gene_annotation,
            to_cpu=to_cpu,
            progress=progress,
        )

    def tidy_scores(
        self,
        scores: list[VariantScore] | list[list[VariantScore]],
        organism: str | int | None = None,
        match_gene_strand: bool = True,
        include_extended_metadata: bool = True,
    ):
        """Convert variant scores to a tidy DataFrame using loaded metadata.
        
        This is a convenience wrapper around `alphagenome_pytorch.variant_scoring.types.tidy_scores`
        that automatically injects the metadata loaded into this model.

        Args:
            scores: Scoring output
            organism: Organism to use for metadata lookup. Uses default if None.
            match_gene_strand: Filter strand mismatches
            include_extended_metadata: Include extra metadata columns
        
        Returns:
            pd.DataFrame
        """
        # Resolve organism for metadata
        # (This handles the case where user might want to tidy mouse scores using mouse metadata)
        idx = self._resolve_organism_index(organism)
        metadata = self._track_metadata.get(idx, {})
        
        return tidy_scores(
            scores,
            track_metadata=metadata,
            match_gene_strand=match_gene_strand,
            include_extended_metadata=include_extended_metadata,
        )

    def ism_matrix(
        self,
        variant_scores: list[float],
        variants: list[Variant],
        interval: Interval,
        multiply_by_sequence: bool = True,
        vocabulary: str = 'ACGT',
    ) -> torch.Tensor:
        """Construct ISM contribution matrix with mean-centering.

        Creates a (sequence_length, 4) matrix where each position contains
        the contribution score for mutating the reference base to each
        possible alternate base.

        Args:
            variant_scores: List of scores for each variant (from ISM scoring).
            variants: List of Variant objects corresponding to scores.
            interval: Genomic interval that was scored.
            multiply_by_sequence: If True, zero out the reference base positions
                (since we only have alt scores). Default True.
            vocabulary: Nucleotide vocabulary order. Default 'ACGT'.

        Returns:
            Tensor of shape (interval.width, 4) with mean-centered ISM scores.
            
        Example:
            >>> ism_scores = scorer.score_ism_variants(interval, center, scorers)
            >>> # Flatten to get score per variant
            >>> flat_scores = [s[0].scores.mean().item() for s in ism_scores]
            >>> variants = [...]  # Same order as ism_scores
            >>> matrix = scorer.ism_matrix(flat_scores, variants, interval)
        """
        import numpy as np

        scores = np.zeros((interval.width, len(vocabulary)), dtype=np.float32)
        filled = np.zeros((interval.width, len(vocabulary)), dtype=bool)
        base_index = {base: i for i, base in enumerate(vocabulary)}

        for variant, score in zip(variants, variant_scores):
            if not variant.is_snv:
                continue
            position = variant.start - interval.start
            if 0 <= position < interval.width:
                alt_base = variant.alternate_bases.upper()
                if alt_base in base_index:
                    scores[position, base_index[alt_base]] = score
                    filled[position, base_index[alt_base]] = True

        # Mean-center across alternatives (excluding reference)
        # For each position, subtract mean of filled values
        for pos in range(interval.width):
            n_filled = filled[pos].sum()
            if n_filled > 0:
                mean_score = scores[pos, filled[pos]].mean()
                scores[pos] -= mean_score / (len(vocabulary) - 1)

        if multiply_by_sequence:
            # Zero out positions where we don't have data
            scores = scores * (~filled).astype(np.float32)

        return torch.from_numpy(scores)

    def close(self):
        """Close any open file handles."""
        if self._fasta_extractor is not None:
            self._fasta_extractor.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Recommended scorer presets matching JAX reference
RECOMMENDED_VARIANT_SCORERS = {
    'ATAC': 'CenterMaskScorer(output=atac, width=501, agg=diff_log2_sum)',
    'DNASE': 'CenterMaskScorer(output=dnase, width=501, agg=diff_log2_sum)',
    'CHIP_TF': 'CenterMaskScorer(output=chip_tf, width=501, agg=diff_log2_sum)',
    'CHIP_HISTONE': 'CenterMaskScorer(output=chip_histone, width=2001, agg=diff_log2_sum)',
    'CAGE': 'CenterMaskScorer(output=cage, width=501, agg=diff_log2_sum)',
    'PROCAP': 'CenterMaskScorer(output=procap, width=501, agg=diff_log2_sum)',
    'CONTACT_MAPS': 'ContactMapScorer()',
    'RNA_SEQ': 'GeneMaskLFCScorer(output=rna_seq)',
    'RNA_SEQ_ACTIVE': 'GeneMaskActiveScorer(output=rna_seq)',
    'SPLICE_SITES': 'GeneMaskSplicingScorer(output=splice_sites, width=None)',
    'SPLICE_SITE_USAGE': 'GeneMaskSplicingScorer(output=splice_site_usage, width=None)',
    'SPLICE_JUNCTIONS': 'SpliceJunctionScorer()',
    'POLYADENYLATION': 'PolyadenylationScorer()',
}


def get_recommended_scorers(organism: str = 'human') -> list[BaseVariantScorer]:
    """Get recommended variant scorers for an organism.

    Args:
        organism: 'human' or 'mouse'

    Returns:
        List of recommended scorer instances
    """
    from .scorers import (
        CenterMaskScorer,
        ContactMapScorer,
        GeneMaskActiveScorer,
        GeneMaskLFCScorer,
        GeneMaskSplicingScorer,
        PolyadenylationScorer,
        SpliceJunctionScorer,
    )
    from .types import AggregationType, OutputType

    scorers = [
        # Chromatin accessibility
        CenterMaskScorer(OutputType.ATAC, 501, AggregationType.DIFF_LOG2_SUM),
        CenterMaskScorer(OutputType.DNASE, 501, AggregationType.DIFF_LOG2_SUM),
        # TF binding
        CenterMaskScorer(OutputType.CHIP_TF, 501, AggregationType.DIFF_LOG2_SUM),
        # Histone modifications (wider window)
        CenterMaskScorer(OutputType.CHIP_HISTONE, 2001, AggregationType.DIFF_LOG2_SUM),
        # Gene expression
        CenterMaskScorer(OutputType.CAGE, 501, AggregationType.DIFF_LOG2_SUM),
        CenterMaskScorer(OutputType.PROCAP, 501, AggregationType.DIFF_LOG2_SUM),
        # 3D chromatin
        ContactMapScorer(),
        # Gene-level expression
        GeneMaskLFCScorer(OutputType.RNA_SEQ),
        GeneMaskActiveScorer(OutputType.RNA_SEQ),
        # Splicing
        GeneMaskSplicingScorer(OutputType.SPLICE_SITES, width=None),
        GeneMaskSplicingScorer(OutputType.SPLICE_SITE_USAGE, width=None),
        SpliceJunctionScorer(),
        # Active versions (non-directional)
        CenterMaskScorer(OutputType.ATAC, 501, AggregationType.ACTIVE_SUM),
        CenterMaskScorer(OutputType.DNASE, 501, AggregationType.ACTIVE_SUM),
        CenterMaskScorer(OutputType.CHIP_TF, 501, AggregationType.ACTIVE_SUM),
        CenterMaskScorer(OutputType.CHIP_HISTONE, 2001, AggregationType.ACTIVE_SUM),
        CenterMaskScorer(OutputType.CAGE, 501, AggregationType.ACTIVE_SUM),
        CenterMaskScorer(OutputType.PROCAP, 501, AggregationType.ACTIVE_SUM),
    ]

    # Add polyadenylation scorer only for human
    if organism.lower() == 'human':
        scorers.append(PolyadenylationScorer())

    return scorers

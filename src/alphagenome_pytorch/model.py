from typing import Optional, Tuple, Union
from pathlib import Path
import warnings

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from . import layers, convolutions, attention, embeddings, heads
from .config import DtypePolicy
from alphagenome_pytorch.utils.splicing import generate_splice_site_positions

class SequenceEncoder(nn.Module):
    """Encodes DNA sequence to trunk representation. Outputs NCL format (B, C, S)."""

    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = False
        self.dna_embedder = convolutions.DnaEmbedder()
        self.pool = layers.Pool1d(kernel_size=2)

        self.down_blocks = nn.ModuleList()
        in_channels = 768  # Initial output from embedder

        # 6 blocks: bin sizes 2, 4, 8, 16, 32, 64
        self.bin_sizes = [2, 4, 8, 16, 32, 64]
        for _ in self.bin_sizes:
            self.down_blocks.append(convolutions.DownResBlock(in_channels))
            in_channels += 128

    def forward(self, x):
        # x input: (B, S, 4) from user - NLC format
        x = x.transpose(1, 2)  # → (B, 4, S) NCL format

        intermediates = {}
        x = self.dna_embedder(x)
        intermediates['bin_size_1'] = x
        x = self.pool(x)

        for i, block in enumerate(self.down_blocks):
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            bin_size = self.bin_sizes[i]
            intermediates[f'bin_size_{bin_size}'] = x
            x = self.pool(x)

        # x: (B, 1536, 1024), intermediates: all NCL
        return x, intermediates

class SequenceDecoder(nn.Module):
    """Decodes trunk to full resolution. Operates on NCL format (B, C, S)."""

    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = False

        # bin sizes: 64, 32, 16, 8, 4, 2, 1
        self.bin_sizes = [64, 32, 16, 8, 4, 2, 1]

        # Channel sizes from encoder:
        # 1: 768, 2: 896, 4: 1024, 8: 1152, 16: 1280, 32: 1408, 64: 1536
        self.up_blocks = nn.ModuleList()
        current_channels = 1536

        skip_channels_map = {
            64: 1536, 32: 1408, 16: 1280, 8: 1152, 4: 1024, 2: 896, 1: 768
        }

        for bin_size in self.bin_sizes:
            skip_ch = skip_channels_map[bin_size]
            self.up_blocks.append(convolutions.UpResBlock(
                in_channels=current_channels, skip_channels=skip_ch
            ))
            current_channels = skip_ch

    def forward(self, x, intermediates):
        # x: (B, C, S) - NCL format
        for i, bin_size in enumerate(self.bin_sizes):
            block = self.up_blocks[i]
            skip = intermediates.pop(f'bin_size_{bin_size}')
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x = checkpoint(block, x, skip, use_reentrant=False)
            else:
                x = block(x, skip)
            del skip
        return x  # (B, 768, S) - NCL format

class TransformerTower(nn.Module):
    """Transformer tower. Operates on NLC format (B, S, C) - native for attention."""

    def __init__(self, d_model):
        super().__init__()
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList()
        # 9 blocks
        for i in range(9):
            is_even = (i % 2 == 0)
            pair_update = attention.PairUpdateBlock(d_model) if is_even else None

            # Layer components
            attn_bias = attention.AttentionBiasBlock(pair_dim=128)
            mha = attention.MHABlock(d_model)
            mlp = attention.MLPBlock(d_model)

            self.blocks.append(nn.ModuleDict({
                'pair_update': pair_update,
                'attn_bias': attn_bias,
                'mha': mha,
                'mlp': mlp
            }))

    def _forward_block(self, block, x, pair_x, compute_dtype):
        if block['pair_update'] is not None:
            pair_x = block['pair_update'](x, pair_x, compute_dtype=compute_dtype)
        mha_bias = block['attn_bias'](pair_x)
        x = x + block['mha'](x, mha_bias, compute_dtype=compute_dtype)
        x = x + block['mlp'](x)
        return x, pair_x

    def forward(self, x, compute_dtype=None):
        # x: (B, S, D)
        pair_x = None

        for block in self.blocks:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x, pair_x = checkpoint(
                    self._forward_block, block, x, pair_x, compute_dtype,
                    use_reentrant=False,
                )
            else:
                x, pair_x = self._forward_block(block, x, pair_x, compute_dtype)

        return x, pair_x

class AlphaGenome(nn.Module):
    """Main AlphaGenome model for genomic sequence analysis.

    Matches JAX: alphagenome_research.model.model.AlphaGenome

    The model predicts various genomic tracks (ATAC, DNASE, CAGE, etc.) and
    contact maps from DNA sequences.

    Note that by default we include the splicing heads.

    Args:
        num_organisms: Number of organisms (default 2: human, mouse).
        dtype_policy: DtypePolicy controlling precision for params/compute/output.
                      Use DtypePolicy.full_float32() for safe defaults (works everywhere).
                      Use DtypePolicy.mixed_precision() for JAX-matching bfloat16 compute.
                      Defaults to DtypePolicy.full_float32() if not specified.
        track_means_dict: Optional dict mapping head names to track_means tensors.
                          Not needed if loading weights from convert_weights.py, which
                          bundles track_means into the weights file.
        gradient_checkpointing: If True, enable gradient checkpointing in the
                                encoder and decoder to reduce memory usage during
                                training at the cost of additional compute.

    Example:
        from alphagenome_pytorch import AlphaGenome
        from alphagenome_pytorch.config import DtypePolicy

        # Load pretrained model (recommended):
        model = AlphaGenome.from_pretrained('model.pth', device='cuda')

        # Or using load_state_dict directly:
        model = AlphaGenome()
        model.load_state_dict(torch.load('model.pth', weights_only=True))
        model.cuda()

        # JAX-matching mixed precision (bfloat16 compute):
        model = AlphaGenome.from_pretrained(
            'model.pth',
            dtype_policy=DtypePolicy.mixed_precision(),
        )
    """
    def __init__(
        self,
        num_organisms: int = 2,
        dtype_policy: Optional[DtypePolicy] = None,
        track_means_dict: Optional[dict] = None,
        gradient_checkpointing: bool = False,
    ):
        """Initialize AlphaGenome model."""
        super().__init__()
        self.num_organisms = num_organisms
        track_means_dict = track_means_dict or {}

        # Set dtype policy (default: full float32, works everywhere)
        self.dtype_policy = dtype_policy if dtype_policy is not None else DtypePolicy.default()

        self.encoder = SequenceEncoder()
        self.encoder.gradient_checkpointing = gradient_checkpointing

        # Architecture dimension constants
        TRUNK_DIM = 1536           # Encoder output / transformer dimension
        EMBEDDING_128BP_DIM = 3072 # 128bp output embedder dimension
        DECODER_DIM = 768          # Decoder output / 1bp embedder input dimension

        self.organism_embed = nn.Embedding(num_organisms, TRUNK_DIM)
        
        self.tower = TransformerTower(d_model=TRUNK_DIM)
        self.tower.gradient_checkpointing = gradient_checkpointing
        self.decoder = SequenceDecoder()
        self.decoder.gradient_checkpointing = gradient_checkpointing

        # Output Embedders - all NCL format
        # Trunk (1536) -> 128bp Embeddings (3072)
        self.embedder_128bp = embeddings.OutputEmbedder(
            in_channels=TRUNK_DIM,
            out_channels=EMBEDDING_128BP_DIM,
            num_organisms=num_organisms,
        )

        # Decoder (768) + Skip (3072) -> 1bp Embeddings (1536)
        self.embedder_1bp = embeddings.OutputEmbedder(
            in_channels=DECODER_DIM,
            out_channels=TRUNK_DIM,
            num_organisms=num_organisms,
        )
        # Skip projection: Conv1d for NCL format
        self.embedder_1bp.project_skip = nn.Conv1d(
            EMBEDDING_128BP_DIM,
            TRUNK_DIM,
            kernel_size=1,
            bias=False,
        )
        
        # Pair Embedder
        self.embedder_pair = embeddings.OutputPair(dim=128, num_organisms=num_organisms)
        
        # Heads
        # We replace placeholders with specific named heads matching JAX reference.
        # Resolutions: Most use [1, 128], some use only [128].
        
        self.heads = nn.ModuleDict()

        # Embedding dimensions per resolution (single source of truth)
        _EMBEDDING_DIMS = {1: TRUNK_DIM, 128: EMBEDDING_128BP_DIM}

        # Helper to simplify head creation
        def add_head(name, num_tracks, resolutions=(1, 128), apply_squashing=False):
            # Get track_means from dict if provided
            track_means = track_means_dict.get(name, None)
            self.heads[name] = heads.GenomeTracksHead(
                in_channels=_EMBEDDING_DIMS,
                num_tracks=num_tracks,
                resolutions=resolutions,
                num_organisms=num_organisms,
                apply_squashing=apply_squashing,
                track_means=track_means,
            )

        # Standard Heads Config (from reference heads.py)
        # apply_squashing is True only for RNA_SEQ
        add_head('atac', 256, [1, 128], apply_squashing=False)
        add_head('dnase', 384, [1, 128], apply_squashing=False)
        add_head('procap', 128, [1, 128], apply_squashing=False)
        add_head('cage', 640, [1, 128], apply_squashing=False)
        add_head('rna_seq', 768, [1, 128], apply_squashing=True)
        add_head('chip_tf', 1664, [128], apply_squashing=False)
        add_head('chip_histone', 1152, [128], apply_squashing=False)

        # Contact Maps Head (from pair embeddings)
        self.contact_maps_head = heads.ContactMapsHead(
            in_features=heads.PAIR_EMBEDDING_DIM,
            num_tracks=heads.CONTACT_MAPS_OUTPUT_TRACKS,
            num_organisms=num_organisms,
        )

        # Splice heads have different structure (Single resolution usually or specific logic)
        # reference: SpliceSitesClassificationHead, SpliceSitesUsageHead, SpliceSitesJunctionHead
        if num_organisms == 1:
            splice_usage_tracks_per_organism = (734,)
            splice_junction_tracks_per_organism = (367,)
        elif num_organisms == 2:
            # Human/mouse defaults matching the current pretrained setup.
            splice_usage_tracks_per_organism = (734, 180)
            splice_junction_tracks_per_organism = (367, 90)
        else:
            warnings.warn(
                "AlphaGenome currently only supports num_organisms in {1, 2}. "
                "For now, splicing heads use hardcoded human/mouse track configs."
            )

        self.splice_sites_classification_head = heads.SpliceSitesClassificationHead(
            in_channels=TRUNK_DIM, num_organisms=num_organisms
        )
        self.splice_sites_usage_head = heads.SpliceSitesUsageHead(
            in_channels=TRUNK_DIM,
            num_output_tracks=max(splice_usage_tracks_per_organism),
            num_organisms=num_organisms,
            num_tracks_per_organism=splice_usage_tracks_per_organism,
        )
        self.splice_sites_junction_head = heads.SpliceSitesJunctionHead(
            in_channels=TRUNK_DIM,
            hidden_dim=DECODER_DIM,
            num_tissues=max(splice_junction_tracks_per_organism),
            num_organisms=num_organisms,
            num_tracks_per_organism=splice_junction_tracks_per_organism,
        )

        # Convert model parameters to params_dtype
        # JAX keeps params in float32 even when computing in bfloat16
        if self.dtype_policy.params_dtype != torch.float32:
            self.to(self.dtype_policy.params_dtype)

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        dtype_policy: Optional[DtypePolicy] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> "AlphaGenome":
        """Load a pretrained AlphaGenome model from a weights file.

        Args:
            path: Path to the weights file (.pth) created by convert_weights.py.
            dtype_policy: DtypePolicy for precision control. Defaults to DtypePolicy.full_float32().
                          Use DtypePolicy.mixed_precision() for JAX-matching bfloat16 compute.
            device: Device to load the model onto ('cuda', 'cpu', etc.). If None, loads to CPU.
            **kwargs: Additional arguments passed to AlphaGenome constructor
                      (e.g., num_organisms, gradient_checkpointing).

        Returns:
            AlphaGenome model with loaded weights.

        Note:
            This method is backward compatible with older weights files that don't
            include track_means buffers. If track_means are missing, a warning is
            issued and default values (zeros) are used. For proper output scaling
            with older weights, load track_means separately using load_track_means().

        Example:
            # Load model to GPU with default settings (float32):
            model = AlphaGenome.from_pretrained('model.pth', device='cuda')

            # Load with JAX-matching mixed precision (bfloat16 compute):
            model = AlphaGenome.from_pretrained(
                'model.pth',
                dtype_policy=DtypePolicy.mixed_precision(),
                device='cuda:0',
            )
        """
        if dtype_policy is None:
            dtype_policy = DtypePolicy.default()

        # Create model and move to target device first for efficient weight loading.
        # This allows loading state_dict directly to the target device, avoiding
        # cross-device transfers during load_state_dict.
        model = cls(dtype_policy=dtype_policy, **kwargs)
        if device:
            model.to(device)

        # Load state dict directly to the device where model lives
        map_location = device if device else 'cpu'
        if Path(path).suffix == '.safetensors':
            from safetensors.torch import load_file
            state_dict = load_file(path, device=str(map_location))
        else:
            state_dict = torch.load(path, map_location=map_location, weights_only=True)

        # Use strict=False to allow loading older weights without track_means,
        # but validate the result to catch other issues
        result = model.load_state_dict(state_dict, strict=False)

        # Check for unexpected keys (always an error - indicates architecture mismatch)
        if result.unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys in state_dict: {result.unexpected_keys}. "
                "This may indicate a version mismatch between the weights file "
                "and the model architecture."
            )

        # Check for missing keys - allow track_means but warn, error on others
        if result.missing_keys:
            track_means_keys = [k for k in result.missing_keys if 'track_means' in k]
            other_missing = [k for k in result.missing_keys if 'track_means' not in k]

            if other_missing:
                raise RuntimeError(
                    f"Missing keys in state_dict: {other_missing}. "
                    "This may indicate a version mismatch between the weights file "
                    "and the model architecture."
                )

            if track_means_keys:
                warnings.warn(
                    f"Weights file is missing track_means buffers ({len(track_means_keys)} keys). "
                    "Using default values (zeros). For proper output scaling, either: "
                    "(1) use a newer weights file with bundled track_means, or "
                    "(2) load track_means separately using model.load_track_means().",
                    UserWarning,
                    stacklevel=2,
                )

        return model

    @classmethod
    def from_delta(
        cls,
        delta_path: Union[str, Path],
        base_path: Union[str, Path],
        dtype_policy: Optional[DtypePolicy] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> "AlphaGenome":
        """Load a finetuned AlphaGenome model from delta weights and base weights.

        This is the simplest way to load a finetuned model. It reconstructs the
        full model from a small delta weights file (adapters + heads) and the
        base pretrained weights.

        Args:
            delta_path: Path to the delta weights file (.safetensors or .pth)
                created by ``export_delta_weights()``.
            base_path: Path to the base pretrained weights file (.pth or
                .safetensors) created by ``convert_weights.py``.
            dtype_policy: DtypePolicy for precision control. Defaults to
                DtypePolicy.full_float32().
            device: Device to load the model onto ('cuda', 'cpu', etc.).
                If None, loads to CPU.
            **kwargs: Additional arguments passed to AlphaGenome constructor
                (e.g., num_organisms, gradient_checkpointing).

        Returns:
            AlphaGenome model with base weights, adapters, and finetuned heads.

        Example:
            >>> model = AlphaGenome.from_delta(
            ...     'colleague_lora.safetensors',
            ...     'alphagenome_pretrained.pth',
            ...     device='cuda',
            ... )
        """
        from alphagenome_pytorch.extensions.finetuning.checkpointing import (
            load_delta_config,
            load_delta_weights,
        )
        from alphagenome_pytorch.extensions.finetuning.transfer import (
            load_trunk,
            prepare_for_transfer,
        )

        if dtype_policy is None:
            dtype_policy = DtypePolicy.default()

        # 1. Create base model and load pretrained trunk
        model = cls(dtype_policy=dtype_policy, **kwargs)
        model = load_trunk(model, base_path, exclude_heads=True)

        # 2. Read config and set up adapters/heads
        config = load_delta_config(delta_path)
        model = prepare_for_transfer(model, config)

        # 3. Load delta weights
        load_delta_weights(model, delta_path)

        # 4. Move to target device
        if device:
            model.to(device)

        return model

    def _compute_embeddings_ncl(self, dna_sequence, organism_index, resolutions=None):
        """Internal method to compute embeddings in NCL format.

        Returns:
            Tuple of (embeddings_1bp, embeddings_128bp, embeddings_pair, need_1bp)
            where sequence embeddings are in NCL format (B, C, S).
        """
        # Cast input to compute dtype
        dna_sequence = self.dtype_policy.cast_to_compute(dna_sequence)

        # ===== ENCODER (NCL) =====
        trunk, intermediates = self.encoder(dna_sequence)  # trunk: (B, 1536, 1024)

        # ===== NCL → NLC for Transformer =====
        trunk = trunk.transpose(1, 2)  # → (B, 1024, 1536)

        # Add organism embedding (NLC format)
        org_emb = self.organism_embed(organism_index).unsqueeze(1)  # (B, 1, 1536)
        trunk = trunk + org_emb

        # ===== TRANSFORMER (NLC) =====
        trunk, pair_activations = self.tower(trunk, compute_dtype=self.dtype_policy.compute_dtype)
        # trunk: (B, 1024, 1536) NLC

        # ===== NLC → NCL for Decoder/Embedders =====
        trunk_ncl = trunk.transpose(1, 2)  # → (B, 1536, 1024)

        # Determine which resolutions are needed
        need_1bp = resolutions is None or 1 in resolutions

        # ===== OUTPUT EMBEDDINGS (NCL format) =====
        # 128bp Embeddings - always computed
        embeddings_128bp = self.embedder_128bp(
            trunk_ncl, organism_index, channels_last=False
        )  # (B, 3072, 1024)

        # 1bp Embeddings (from decoder + skip) - skip if not needed
        if need_1bp:
            decoded_x = self.decoder(trunk_ncl, intermediates)  # (B, 768, 131072)
            embeddings_1bp = self.embedder_1bp(
                decoded_x, organism_index, skip_x=embeddings_128bp, channels_last=False
            )  # (B, 1536, 131072)
        else:
            embeddings_1bp = None
            del intermediates  # Free memory from encoder skip connections

        # Pair Embeddings (B, S, S, D) - different format, not NCL
        embeddings_pair = self.embedder_pair(pair_activations, organism_index)

        return embeddings_1bp, embeddings_128bp, embeddings_pair, need_1bp

    def encode(
        self,
        dna_sequence,
        organism_index,
        resolutions=None,
        channels_last=True,
    ):
        """Extract embeddings without running prediction heads.

        This method runs the encoder, transformer, decoder, and output embedders
        to produce embeddings that can be used with custom heads for fine-tuning.

        Args:
            dna_sequence: One-hot encoded DNA sequence (B, S, 4) - NLC input format.
            organism_index: Organism index per batch (B,). 0=human, 1=mouse.
            resolutions: Tuple of resolutions to compute, e.g. (1, 128) or (128,).
                         If None, computes all resolutions. When 1bp is not needed,
                         the expensive decoder is skipped for faster computation.
            channels_last: Output format for sequence embeddings.
                - True (default): NLC format (B, S, C) - user-friendly, matches JAX
                - False: NCL format (B, C, S) - efficient for Conv1d heads

        Returns:
            Dict with keys:
                - 'embeddings_1bp': (B, S, 1536) or (B, 1536, S) at 1bp resolution.
                  Only present if 1 is in resolutions (or resolutions is None).
                - 'embeddings_128bp': (B, S//128, 3072) or (B, 3072, S//128) at 128bp.
                - 'embeddings_pair': (B, S//2048, S//2048, 128) pair embeddings.

        Example:
            # Get embeddings for fine-tuning with a custom head
            model = AlphaGenome.from_pretrained('model.pth', device='cuda')
            model.eval()

            # Freeze backbone
            for param in model.parameters():
                param.requires_grad = False

            # Get embeddings (128bp only for efficiency)
            with torch.no_grad():
                emb = model.encode(dna_seq, organism_idx, resolutions=(128,))

            # Use with custom head (NCL format for Conv1d)
            emb = model.encode(dna_seq, organism_idx, channels_last=False)
            custom_output = my_conv_head(emb['embeddings_128bp'])
        """
        embeddings_1bp, embeddings_128bp, embeddings_pair, need_1bp = \
            self._compute_embeddings_ncl(dna_sequence, organism_index, resolutions)

        # Build output dict with requested format
        # Use contiguous() after transpose to ensure memory layout is optimal
        # for downstream operations (Conv1d, CUDA kernels, etc.)
        outputs = {}

        if channels_last:
            if need_1bp:
                outputs['embeddings_1bp'] = embeddings_1bp.transpose(1, 2).contiguous()
            outputs['embeddings_128bp'] = embeddings_128bp.transpose(1, 2).contiguous()
        else:
            if need_1bp:
                outputs['embeddings_1bp'] = embeddings_1bp
            outputs['embeddings_128bp'] = embeddings_128bp

        outputs['embeddings_pair'] = embeddings_pair

        return self._cast_outputs(outputs)

    def forward(
        self,
        dna_sequence,
        organism_index,
        *,
        splice_site_positions=None,
        return_embeddings=False,
        return_scaled_predictions=False,
        resolutions=None,
        heads: Optional[Tuple[str, ...]] = None,
        channels_last=True,
        embeddings_only=False,
        encoder_only=False,
    ):
        """Forward pass through the model.

        Args:
            dna_sequence: One-hot encoded DNA sequence (B, S, 4) - NLC input
            organism_index: Organism index per batch (B,). 0=human, 1=mouse.
            splice_site_positions: Optional pre-computed splice site positions
                (B, 4, K). If provided, skips internal Top-K selection.
            return_embeddings: If True, include embeddings in output.
            return_scaled_predictions: If True, return model space (for loss).
                                       If False, return experimental space (for inference).
            resolutions: Tuple of resolutions to compute, e.g. (1, 128) or (128,).
                         If None, computes all resolutions.
            heads: Tuple of head names to compute, e.g. ('atac',) or ('atac', 'dnase').
                   If None, computes all heads. Use this to skip expensive unused heads
                   during inference.
            channels_last: Format for embeddings and head outputs.
                - True (default): NLC format (B, S, C) - user-friendly, matches JAX
                - False: NCL format (B, C, S) - for training efficiency (0 transposes)
            embeddings_only: If True, skip all head computation and only return
                embeddings. Useful for fine-tuning where only a custom head is used.
                Implies return_embeddings=True.
            encoder_only: If True, run only the CNN encoder (skip organism embedding,
                transformer, decoder, and heads). Returns raw encoder output in
                ``{"encoder_output": tensor}`` where tensor has shape (B, S//128, 1536).

        Returns:
            Dict of predictions from each head. Keys are head names
            (atac, dnase, cage, etc.), values are dicts mapping
            resolution (1 or 128) to prediction tensors.
            If return_embeddings is True, also contains 'embeddings_1bp' and 'embeddings_128bp'.
            If encoder_only is True, returns ``{"encoder_output": tensor}`` only.

        Raises:
            ValueError: If unknown head names are specified in ``heads``.
        """
        if encoder_only:
            # Return raw CNN encoder output before organism embedding and transformer.
            trunk, _intermediates = self.encoder(dna_sequence)
            outputs = {"encoder_output": trunk}
            return self._cast_outputs(outputs)

        # Validate heads parameter
        if heads is not None:
            # Build set of all valid head names
            valid_heads = set(self.heads.keys())
            if self.contact_maps_head is not None:
                valid_heads.add('pair_activations')
            if self.splice_sites_classification_head is not None:
                valid_heads.add('splice_sites_classification')
            if self.splice_sites_usage_head is not None:
                valid_heads.add('splice_sites_usage')
            if self.splice_sites_junction_head is not None:
                valid_heads.add('splice_sites_junction')

            # Check for unknown heads
            unknown_heads = set(heads) - valid_heads
            if unknown_heads:
                raise ValueError(
                    f"Unknown head names: {sorted(unknown_heads)}. "
                    f"Available heads: {sorted(valid_heads)}"
                )
            head_set = set(heads)
        else:
            head_set = None

        # Compute embeddings (NCL format internally)
        embeddings_1bp, embeddings_128bp, embeddings_pair, need_1bp = \
            self._compute_embeddings_ncl(dna_sequence, organism_index, resolutions)

        if need_1bp:
            embeddings_dict = {1: embeddings_1bp, 128: embeddings_128bp}
        else:
            embeddings_dict = {128: embeddings_128bp}

        # ===== HEADS =====
        outputs = {}

        if not embeddings_only:
            for name, head in self.heads.items():
                if head_set is not None and name not in head_set:
                    continue
                outputs[name] = head(
                    embeddings_dict, organism_index,
                    return_scaled=return_scaled_predictions,
                    channels_last=channels_last,
                )

            # Contact Maps (pair activations format)
            if self.contact_maps_head is not None:
                if head_set is None or 'pair_activations' in head_set:
                    outputs['pair_activations'] = self.contact_maps_head(
                        embeddings_pair, organism_index, channels_last=channels_last
                    )

            # Splice predictions (require 1bp embeddings)
            need_splice = head_set is None or any(
                k in head_set for k in ('splice_sites_classification', 'splice_sites_usage', 'splice_sites_junction')
            )
            if need_1bp and need_splice:
                # Also compute classification when junction needs it for position generation
                classification_output = None
                need_classification = (
                    head_set is None
                    or 'splice_sites_classification' in head_set
                    or (
                        'splice_sites_junction' in head_set
                        and splice_site_positions is None
                    )
                )
                if self.splice_sites_classification_head is not None and need_classification:
                    classification_output = self.splice_sites_classification_head(
                        embeddings_1bp, organism_index, channels_last=channels_last
                    )
                    if head_set is None or 'splice_sites_classification' in head_set:
                        outputs['splice_sites_classification'] = classification_output
                if self.splice_sites_usage_head is not None:
                    if head_set is None or 'splice_sites_usage' in head_set:
                        outputs['splice_sites_usage'] = self.splice_sites_usage_head(
                            embeddings_1bp, organism_index, channels_last=channels_last
                        )

                if self.splice_sites_junction_head is not None:
                    if head_set is None or 'splice_sites_junction' in head_set:
                        # Use provided positions if given, otherwise generate from classification
                        if splice_site_positions is not None:
                            top_k_positions = splice_site_positions
                        else:
                            if classification_output is None:
                                raise ValueError(
                                    "splice_sites_junction requires either splice_site_positions "
                                    "or an available splice_sites_classification head"
                                )
                            # probs: (B, S, 5) NLC - already correct format for generate_splice_site_positions
                            splice_site_probs = classification_output['probs']

                            # If NCL (channels_last=False), transpose back to NLC for generate_splice_site_positions
                            if not channels_last:
                                splice_site_probs = splice_site_probs.transpose(1, 2)

                            top_k_positions = generate_splice_site_positions(
                                ref=splice_site_probs,
                                alt=None,
                                true_splice_sites=None,
                                k=512,
                                pad_to_length=512,
                                threshold=0.1,
                            )
                        outputs['splice_sites_junction'] = self.splice_sites_junction_head(
                            embeddings_1bp,
                            organism_index,
                            channels_last=channels_last,
                            splice_site_positions=top_k_positions,
                        )

        if return_embeddings or embeddings_only:
            if channels_last:
                if need_1bp:
                    outputs['embeddings_1bp'] = embeddings_1bp.transpose(1, 2).contiguous()
                outputs['embeddings_128bp'] = embeddings_128bp.transpose(1, 2).contiguous()
            else:
                if need_1bp:
                    outputs['embeddings_1bp'] = embeddings_1bp
                outputs['embeddings_128bp'] = embeddings_128bp

        return self._cast_outputs(outputs)

    def _cast_outputs(self, outputs):
        """Recursively cast all output tensors to output_dtype."""
        if torch.is_tensor(outputs):
            return self.dtype_policy.cast_to_output(outputs)
        if isinstance(outputs, dict):
            return {k: self._cast_outputs(v) for k, v in outputs.items()}
        if isinstance(outputs, (list, tuple)):
            casted = [self._cast_outputs(v) for v in outputs]
            return type(outputs)(casted)
        return outputs

    @staticmethod
    def _upcast_outputs(outputs):
        """Recursively upcast low-precision floating-point tensors to float32.

        Matches JAX's tensor_utils.upcast_floating: only upcasts floating-point
        types smaller than float32 (bfloat16, float16). Leaves int tensors and
        float32+ tensors unchanged.
        """
        if torch.is_tensor(outputs):
            if outputs.is_floating_point() and outputs.dtype in (torch.bfloat16, torch.float16):
                return outputs.float()
            return outputs
        if isinstance(outputs, dict):
            return {k: AlphaGenome._upcast_outputs(v) for k, v in outputs.items()}
        if isinstance(outputs, (list, tuple)):
            upcasted = [AlphaGenome._upcast_outputs(v) for v in outputs]
            return type(outputs)(upcasted)
        return outputs

    @torch.no_grad()
    def predict(
        self,
        dna_sequence: torch.Tensor,
        organism_index: Union[torch.Tensor, int],
        **kwargs,
    ) -> dict:
        """Inference-mode forward pass with automatic dtype handling.

        Wraps forward() with:
        - torch.no_grad() for memory efficiency
        - torch.autocast for mixed-precision compute
        - Float32 upcasting of all outputs

        This matches the JAX reference's inference behavior, where outputs are
        upcast to float32 via _upcast_single_batch_predictions before being
        returned to the user.

        Args:
            dna_sequence: One-hot encoded DNA sequence (B, S, 4). Can be any
                float dtype — autocast handles weight/input casting.
            organism_index: Organism index per batch (B,). 0=human, 1=mouse.
            **kwargs: Additional arguments passed to forward()
                (e.g., return_embeddings, resolutions).

        Returns:
            Dict of predictions with all floating-point tensors in float32.
        """
        device_type = "cuda" if dna_sequence.is_cuda else "cpu"
        use_amp = self.dtype_policy.compute_dtype != torch.float32

        # Handle integer organism_index by converting to tensor of shape (B,)
        # forward() expects a tensor for embedding lookups.
        if isinstance(organism_index, int):
            batch_size = dna_sequence.shape[0]
            organism_index = torch.full(
                (batch_size,),
                organism_index,
                dtype=torch.long,
                device=dna_sequence.device
            )

        with torch.autocast(device_type=device_type, dtype=self.dtype_policy.compute_dtype, enabled=use_amp):
            outputs = self.forward(dna_sequence, organism_index, **kwargs)

        return self._upcast_outputs(outputs)

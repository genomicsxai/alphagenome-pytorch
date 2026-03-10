"""Backward pass tests for AlphaGenome PyTorch model.

These tests verify that the model is trainable by checking:
1. All parameters receive gradients
2. Gradients are numerically stable (no NaN/Inf)
3. All heads contribute gradients to the backbone
4. An optimizer step actually updates parameters
"""

import pytest
import torch
import torch.nn.functional as F

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy
from alphagenome_pytorch.model import SequenceEncoder, SequenceDecoder, TransformerTower
from alphagenome_pytorch.convolutions import StandardizedConv1d, ConvBlock, DownResBlock, UpResBlock
from alphagenome_pytorch.attention import MHABlock, MLPBlock, PairUpdateBlock, SequenceToPairBlock
from alphagenome_pytorch.heads import (
    MultiOrganismLinear,
    GenomeTracksHead,
    ContactMapsHead,
    SpliceSitesClassificationHead,
    SpliceSitesUsageHead,
    SpliceSitesJunctionHead,
)
from alphagenome_pytorch.layers import RMSBatchNorm, LayerNorm

from ..gradient_utils import (
    check_all_gradients,
    assert_all_params_have_gradients,
    assert_no_nan_inf_gradients,
    compute_combined_loss,
    compute_head_loss,
)


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Clean GPU memory before and after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Full Model Backward Tests (MOST IMPORTANT)
# =============================================================================


@pytest.mark.integration
class TestFullModelBackward:
    """Test that the full AlphaGenome model supports backward pass."""

    @pytest.fixture
    def small_model(self):
        """Create a model for testing (no pretrained weights needed)."""
        model = AlphaGenome(num_organisms=2, dtype_policy=DtypePolicy.full_float32())
        model.train()
        return model

    @pytest.fixture
    def small_input(self):
        """Create small input for memory-efficient testing."""
        # Use 16384 bp for component tests (smaller for faster tests)
        B, S = 1, 16384
        x = torch.randn(B, S, 4, requires_grad=True)
        org_idx = torch.tensor([0])
        return x, org_idx

    def test_full_model_all_params_receive_gradients(self, small_model, small_input):
        """CRITICAL: Verify EVERY parameter in the model receives a gradient.

        Note: splice_sites_junction_head parameters are excluded because they depend
        on positions from generate_splice_site_positions with a threshold. With random
        weights, the classification head may not produce confident predictions, causing
        all positions to be invalid (-1), which masks out predictions. These parameters
        are tested separately in TestSpliceHeadGradients.test_splice_junction_head_backward.
        """
        model = small_model
        x, org_idx = small_input

        outputs = model(x, org_idx)
        # Include splice heads in loss so they receive gradients
        loss = compute_combined_loss(outputs, include_splice=True)
        loss.backward()

        # Only exclude splice_sites_junction_head params (see docstring)
        # Other splice heads (classification, usage) now included in loss
        exclude_patterns = ["splice_sites_junction_head"]

        # Check EVERY parameter (except excluded)
        params_without_grad = []
        for name, param in model.named_parameters():
            if any(pat in name for pat in exclude_patterns):
                continue
            if param.grad is None or param.grad.norm() == 0:
                params_without_grad.append(name)

        assert len(params_without_grad) == 0, (
            f"{len(params_without_grad)} parameters have no gradient:\n"
            + "\n".join(params_without_grad[:20])
        )

    def test_full_model_no_nan_inf(self, small_model, small_input):
        """Verify no NaN or Inf in any gradient."""
        model = small_model
        x, org_idx = small_input

        outputs = model(x, org_idx)
        loss = compute_combined_loss(outputs)
        loss.backward()

        assert_no_nan_inf_gradients(model)

    def test_optimizer_step_works(self, small_model, small_input):
        """Verify we can actually take an optimizer step (model is trainable)."""
        model = small_model
        x, org_idx = small_input

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Store original params (detached clones)
        original_params = {n: p.detach().clone() for n, p in model.named_parameters()}

        # Forward + backward + step
        outputs = model(x, org_idx)
        loss = compute_combined_loss(outputs)
        loss.backward()
        optimizer.step()

        # Verify params changed
        params_changed = 0
        for name, param in model.named_parameters():
            if not torch.equal(param.data, original_params[name]):
                params_changed += 1

        assert params_changed > 0, "No parameters were updated by optimizer step"

    def test_input_receives_gradient(self, small_model, small_input):
        """Verify gradients flow all the way back to the input."""
        model = small_model
        x, org_idx = small_input

        outputs = model(x, org_idx)
        loss = compute_combined_loss(outputs)
        loss.backward()

        assert x.grad is not None, "Input did not receive gradient"
        assert x.grad.norm() > 0, "Input gradient is zero"
        assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"

    def test_no_nans_in_forward_pass(self, pytorch_model, random_dna_sequence):
        """Verify no NaNs in model predictions (forward pass).

        This test ensures numerical stability in the forward pass by checking
        that all predictions are free of NaN and Inf values. This is critical
        for training stability.

        Merged from test_nans.py which focused on forward pass validation.
        """
        import numpy as np

        device = next(pytorch_model.parameters()).device
        pt_input = torch.tensor(random_dna_sequence).to(device)
        pt_org = torch.tensor([0], dtype=torch.long).to(device)  # Human

        with torch.no_grad():
            outputs = pytorch_model(pt_input, pt_org)

        # Check all output heads for NaN/Inf
        for head_name, head_out in outputs.items():
            if isinstance(head_out, dict):
                # Multi-resolution heads (genomic tracks)
                for res, arr in head_out.items():
                    if arr is not None:
                        assert not torch.isnan(arr).any(), \
                            f"NaN detected in {head_name}/{res} predictions"
                        assert not torch.isinf(arr).any(), \
                            f"Inf detected in {head_name}/{res} predictions"
            elif head_out is not None:
                # Single output heads (contact maps, etc.)
                assert not torch.isnan(head_out).any(), \
                    f"NaN detected in {head_name} predictions"
                assert not torch.isinf(head_out).any(), \
                    f"Inf detected in {head_name} predictions"


# =============================================================================
# All Heads Contribute Gradients Tests
# =============================================================================


@pytest.mark.integration
class TestAllHeadsContributeGradients:
    """Verify gradients flow from EVERY output head to the backbone."""

    @pytest.fixture
    def model_and_input(self):
        """Create model and input for testing."""
        model = AlphaGenome(num_organisms=2, dtype_policy=DtypePolicy.full_float32())
        model.train()

        B, S = 1, 16384
        x = torch.randn(B, S, 4, requires_grad=True)
        org_idx = torch.tensor([0])

        return model, x, org_idx

    @pytest.mark.parametrize(
        "head_name",
        [
            "atac",
            "dnase",
            "procap",
            "cage",
            "rna_seq",
            "chip_tf",
            "chip_histone",
            "pair_activations",
            "splice_sites_classification",
            "splice_sites_usage",
            # Note: splice_sites_junction is excluded because its positions come from
            # generate_splice_site_positions with a threshold. With random weights,
            # the classification head may not produce confident predictions, causing
            # all positions to be invalid (-1), which masks out predictions and breaks
            # gradient flow. See test_splice_junction_head_backward for direct testing.
        ],
    )
    def test_head_contributes_gradients(self, model_and_input, head_name):
        """Test that gradients from each head reach the encoder."""
        model, x, org_idx = model_and_input

        # Reset gradients
        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()

        outputs = model(x, org_idx)

        # Compute loss from ONLY this head
        loss = compute_head_loss(outputs, head_name)
        loss.backward()

        # Check encoder (shared backbone) receives gradients
        encoder_has_grad = any(
            p.grad is not None and p.grad.norm() > 0 for p in model.encoder.parameters()
        )
        assert encoder_has_grad, f"Encoder got no gradients from {head_name}"


# =============================================================================
# Component-Level Backward Tests
# =============================================================================


@pytest.mark.integration
class TestComponentGradients:
    """Test gradient flow through individual components."""

    def test_standardized_conv1d_backward(self):
        """Test gradient flows through StandardizedConv1d with weight standardization."""
        conv = StandardizedConv1d(64, 96, kernel_size=5, padding="same")
        x = torch.randn(2, 64, 128, requires_grad=True)

        y = conv(x)
        loss = y.mean()
        loss.backward()

        # Verify gradients
        assert x.grad is not None, "Input gradient missing"
        assert conv.weight.grad is not None, "Weight gradient missing"
        assert conv.bias.grad is not None, "Bias gradient missing"
        assert conv.scale.grad is not None, "Scale parameter gradient missing"

        # Check no NaN/Inf
        assert not torch.isnan(x.grad).any(), "NaN in input gradient"
        assert not torch.isnan(conv.weight.grad).any(), "NaN in weight gradient"
        assert not torch.isnan(conv.scale.grad).any(), "NaN in scale gradient"

        # Check gradient magnitude is reasonable
        assert conv.weight.grad.norm() > 1e-12, "Weight gradient too small"
        assert conv.weight.grad.norm() < 1e8, "Weight gradient too large"

    def test_multi_organism_linear_backward(self):
        """Test gradient flows through MultiOrganismLinear with bmm operations."""
        B, S, C_in, C_out = 4, 100, 64, 32
        num_organisms = 2

        layer = MultiOrganismLinear(C_in, C_out, num_organisms)
        x = torch.randn(B, S, C_in, requires_grad=True)
        org_idx = torch.tensor([0, 0, 1, 1])  # Mixed organisms

        y = layer(x, org_idx)
        loss = y.mean()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None, "Input gradient missing"
        assert layer.weight.grad is not None, "Weight gradient missing"
        assert layer.bias.grad is not None, "Bias gradient missing"

        # Check no NaN/Inf
        assert not torch.isnan(layer.weight.grad).any(), "NaN in weight gradient"

        # Check both organism weights receive gradients
        org0_grad = layer.weight.grad[0]
        org1_grad = layer.weight.grad[1]
        assert org0_grad.norm() > 1e-12, "Organism 0 weights not receiving gradient"
        assert org1_grad.norm() > 1e-12, "Organism 1 weights not receiving gradient"

    def test_mha_block_backward(self):
        """Test gradient flows through MHABlock with RoPE and attention."""
        B, S, D = 2, 64, 1536

        mha = MHABlock(D)
        x = torch.randn(B, S, D, requires_grad=True)
        attn_bias = torch.randn(B, 8, S, S)  # 8 heads

        y = mha(x, attn_bias, compute_dtype=torch.float32)
        loss = y.mean()
        loss.backward()

        # Check input gradient
        assert x.grad is not None, "Input gradient missing"
        assert not torch.isnan(x.grad).any(), "NaN in input gradient"

        # Check key projections have gradients
        assert mha.q_proj.weight.grad is not None, "Q projection gradient missing"
        assert mha.k_proj.weight.grad is not None, "K projection gradient missing"
        assert mha.v_proj.weight.grad is not None, "V projection gradient missing"

        # Check norms have gradients
        assert mha.norm.weight.grad is not None, "Norm weight gradient missing"

    def test_rms_batch_norm_backward(self):
        """Test gradient flows through RMSBatchNorm."""
        B, S, D = 2, 100, 64

        norm = RMSBatchNorm(D, channels_last=True)
        norm.train()  # Important: ensure training mode
        x = torch.randn(B, S, D, requires_grad=True)

        y = norm(x)
        loss = y.mean()
        loss.backward()

        assert x.grad is not None, "Input gradient missing"
        assert norm.weight.grad is not None, "Weight gradient missing"
        assert norm.bias.grad is not None, "Bias gradient missing"
        assert not torch.isnan(x.grad).any(), "NaN in input gradient"

    def test_pair_update_block_backward(self):
        """Test gradient flows through PairUpdateBlock."""
        B, S, D = 2, 64, 1536
        pair_dim = 128

        block = PairUpdateBlock(D, pair_dim)
        x = torch.randn(B, S, D, requires_grad=True)
        pair_rep = torch.randn(B, S // 16, S // 16, pair_dim, requires_grad=True)

        out = block(x, pair_rep, compute_dtype=torch.float32)
        loss = out.mean()
        loss.backward()

        # Check gradients
        assert x.grad is not None, "Sequence input gradient missing"
        assert pair_rep.grad is not None, "Pair rep input gradient missing"
        assert not torch.isnan(x.grad).any(), "NaN in sequence gradient"

        # Check internal components have gradients
        for name, param in block.named_parameters():
            assert param.grad is not None, f"PairUpdateBlock {name} missing gradient"

    def test_conv_block_backward(self):
        """Test gradient flows through ConvBlock."""
        B, S, C = 2, 128, 64

        block = ConvBlock(C, C, kernel_size=5)
        x = torch.randn(B, C, S, requires_grad=True)

        y = block(x)
        loss = y.mean()
        loss.backward()

        assert x.grad is not None, "Input gradient missing"
        assert not torch.isnan(x.grad).any(), "NaN in gradient"

    def test_down_res_block_backward(self):
        """Test gradient flows through DownResBlock."""
        B, S, C = 2, 128, 768

        block = DownResBlock(C)
        x = torch.randn(B, C, S, requires_grad=True)

        y = block(x)
        loss = y.mean()
        loss.backward()

        assert x.grad is not None, "Input gradient missing"
        assert not torch.isnan(x.grad).any(), "NaN in gradient"

    def test_up_res_block_backward(self):
        """Test gradient flows through UpResBlock."""
        B, S, C_in, C_skip = 2, 64, 1536, 1408

        block = UpResBlock(C_in, C_skip)
        x = torch.randn(B, C_in, S, requires_grad=True)
        skip = torch.randn(B, C_skip, S * 2, requires_grad=True)

        y = block(x, skip)
        loss = y.mean()
        loss.backward()

        assert x.grad is not None, "Input gradient missing"
        assert skip.grad is not None, "Skip connection gradient missing"
        assert not torch.isnan(x.grad).any(), "NaN in gradient"


# =============================================================================
# Splice Head Backward Tests
# =============================================================================


@pytest.mark.integration
class TestSpliceHeadGradients:
    """Test gradient flow through splice site heads."""

    def test_splice_classification_head_backward(self):
        """Test gradient flows through SpliceSitesClassificationHead."""
        B, S = 2, 1024
        in_channels = 1536

        head = SpliceSitesClassificationHead(in_channels=in_channels)
        # Input NCL format: (B, C, S)
        embeddings = torch.randn(B, in_channels, S, requires_grad=True)
        organism_index = torch.tensor([0, 1])

        output = head(embeddings, organism_index)
        loss = output["logits"].mean()
        loss.backward()

        assert embeddings.grad is not None, "Input gradient missing"
        assert not torch.isnan(embeddings.grad).any(), "NaN in gradient"
        assert head.conv.weight.grad is not None, "Conv weight gradient missing"

    def test_splice_usage_head_backward(self):
        """Test gradient flows through SpliceSitesUsageHead."""
        B, S = 2, 1024
        in_channels = 1536

        head = SpliceSitesUsageHead(in_channels=in_channels, num_output_tracks=28)
        # Input NCL format: (B, C, S)
        embeddings = torch.randn(B, in_channels, S, requires_grad=True)
        organism_index = torch.tensor([0, 1])

        output = head(embeddings, organism_index)
        loss = output["logits"].mean()
        loss.backward()

        assert embeddings.grad is not None, "Input gradient missing"
        assert not torch.isnan(embeddings.grad).any(), "NaN in gradient"

    def test_splice_junction_head_backward(self):
        """Test gradient flows through SpliceSitesJunctionHead with RoPE and einsum."""
        B, S, P = 2, 1024, 64
        in_channels, hidden_dim, num_tissues = 1536, 768, 90

        head = SpliceSitesJunctionHead(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_tissues=num_tissues,
        )

        # Input NCL format: (B, C, S)
        embeddings = torch.randn(B, in_channels, S, requires_grad=True)
        organism_index = torch.tensor([0, 1])

        # Create valid splice site positions
        splice_positions = torch.stack(
            [
                torch.randint(0, S, (B, P)),
                torch.randint(0, S, (B, P)),
                torch.randint(0, S, (B, P)),
                torch.randint(0, S, (B, P)),
            ],
            dim=1,
        )

        output = head(embeddings, organism_index, splice_site_positions=splice_positions)
        loss = output["pred_counts"].sum()
        loss.backward()

        # Verify input gradient
        assert embeddings.grad is not None, "Input gradient missing"
        assert not torch.isnan(embeddings.grad).any(), "NaN in input gradient"

        # Verify RoPE param gradients (critical for training)
        for key in ["pos_donor", "pos_acceptor", "neg_donor", "neg_acceptor"]:
            param = head.rope_params[key]
            assert param.grad is not None, f"{key} RoPE params missing gradient"
            assert not torch.isnan(param.grad).any(), f"{key} RoPE params have NaN gradient"

        # Verify conv layer gradient
        assert head.conv.weight.grad is not None, "Conv weight gradient missing"


# =============================================================================
# Encoder/Decoder Backward Tests
# =============================================================================


@pytest.mark.integration
class TestEncoderDecoderBackward:
    """Test gradient flow through encoder and decoder."""

    def test_encoder_backward(self):
        """Test gradient flows through SequenceEncoder."""
        encoder = SequenceEncoder()

        B, S = 1, 8192  # Reduced size for testing
        x = torch.randn(B, S, 4, requires_grad=True)

        trunk, intermediates = encoder(x)
        loss = trunk.sum()
        loss.backward()

        assert x.grad is not None, "Input gradient missing"
        assert x.grad.norm() > 0, "Input gradient is zero"

        # Verify all encoder params have gradients
        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"Encoder {name} missing gradient"

    def test_decoder_backward(self):
        """Test gradient flows through SequenceDecoder."""
        decoder = SequenceDecoder()

        B, S_128 = 1, 64  # S/128
        trunk = torch.randn(B, 1536, S_128, requires_grad=True)

        # Create mock intermediates with correct shapes
        intermediates = {
            "bin_size_1": torch.randn(B, 768, S_128 * 128, requires_grad=True),
            "bin_size_2": torch.randn(B, 896, S_128 * 64, requires_grad=True),
            "bin_size_4": torch.randn(B, 1024, S_128 * 32, requires_grad=True),
            "bin_size_8": torch.randn(B, 1152, S_128 * 16, requires_grad=True),
            "bin_size_16": torch.randn(B, 1280, S_128 * 8, requires_grad=True),
            "bin_size_32": torch.randn(B, 1408, S_128 * 4, requires_grad=True),
            "bin_size_64": torch.randn(B, 1536, S_128 * 2, requires_grad=True),
        }

        output = decoder(trunk, intermediates)
        loss = output.sum()
        loss.backward()

        assert trunk.grad is not None, "Trunk gradient missing"
        for key, inter in intermediates.items():
            assert inter.grad is not None, f"Intermediate {key} gradient missing"

        # Verify all decoder params have gradients
        for name, param in decoder.named_parameters():
            assert param.grad is not None, f"Decoder {name} missing gradient"

    def test_encoder_decoder_roundtrip(self):
        """Test gradient flow through full encoder-decoder path."""
        encoder = SequenceEncoder()
        decoder = SequenceDecoder()

        B, S = 1, 8192
        x = torch.randn(B, S, 4, requires_grad=True)

        trunk, intermediates = encoder(x)
        output = decoder(trunk, intermediates)

        loss = output.sum()
        loss.backward()

        # Verify gradients reach input
        assert x.grad is not None, "Input gradient missing"
        assert x.grad.norm() > 0, "Input gradient is zero"


# =============================================================================
# Transformer Tower Backward Tests
# =============================================================================


@pytest.mark.integration
class TestTransformerTowerBackward:
    """Test gradient flow through transformer tower."""

    def test_transformer_tower_backward(self):
        """Test gradient flow through all 9 transformer layers."""
        tower = TransformerTower(d_model=1536)

        B, S = 1, 64  # Reduced sequence length at 128bp resolution
        x = torch.randn(B, S, 1536, requires_grad=True)

        trunk, pair_activations = tower(x)

        # Loss from both trunk and pair
        loss = trunk.sum() + pair_activations.sum()
        loss.backward()

        # Check input gradient
        assert x.grad is not None, "Input gradient missing"
        assert not torch.isnan(x.grad).any(), "NaN in input gradient"

        # Every block should have gradients
        for i, block in enumerate(tower.blocks):
            for name, param in block.named_parameters():
                assert param.grad is not None, f"Block {i} {name} missing gradient"

    def test_transformer_pair_activations_gradient(self):
        """Test that pair activations receive proper gradients."""
        tower = TransformerTower(d_model=1536)

        B, S = 1, 64
        x = torch.randn(B, S, 1536, requires_grad=True)

        trunk, pair_activations = tower(x)

        # Loss only from pair activations
        loss = pair_activations.sum()
        loss.backward()

        # Input should still receive gradient
        assert x.grad is not None, "Input gradient missing from pair path"
        assert x.grad.norm() > 0, "Input gradient is zero from pair path"


# =============================================================================
# Genome Tracks Head Backward Tests
# =============================================================================


@pytest.mark.integration
class TestGenomeTracksHeadBackward:
    """Test gradient flow through genome tracks heads."""

    def test_genome_tracks_head_backward(self):
        """Test gradient flows through GenomeTracksHead."""
        head = GenomeTracksHead(
            in_channels=None,
            num_tracks=256,
            resolutions=[1, 128],
            num_organisms=2,
        )

        B, S_1bp, S_128bp = 2, 8192, 64
        # Input NCL format: (B, C, S)
        embeddings = {
            1: torch.randn(B, 1536, S_1bp, requires_grad=True),
            128: torch.randn(B, 3072, S_128bp, requires_grad=True),
        }
        organism_index = torch.tensor([0, 1])

        outputs = head(embeddings, organism_index)

        # Compute loss from both resolutions
        loss = outputs[1].mean() + outputs[128].mean()
        loss.backward()

        # Check embeddings receive gradients
        assert embeddings[1].grad is not None, "1bp embeddings gradient missing"
        assert embeddings[128].grad is not None, "128bp embeddings gradient missing"

        # Check head parameters have gradients
        for name, param in head.named_parameters():
            assert param.grad is not None, f"Head {name} missing gradient"

    def test_contact_maps_head_backward(self):
        """Test gradient flows through ContactMapsHead."""
        head = ContactMapsHead(in_features=128, num_tracks=28, num_organisms=2)

        B, S = 2, 64
        pair_embeddings = torch.randn(B, S, S, 128, requires_grad=True)
        organism_index = torch.tensor([0, 1])

        output = head(pair_embeddings, organism_index)
        loss = output.mean()
        loss.backward()

        assert pair_embeddings.grad is not None, "Pair embeddings gradient missing"
        assert not torch.isnan(pair_embeddings.grad).any(), "NaN in gradient"

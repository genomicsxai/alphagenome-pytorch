"""Unit tests for unified transform module.

Tests all transform types to ensure correct shape transformations
for both weight conversion and gradient alignment.
"""

import pytest
import numpy as np
from alphagenome_pytorch.jax_compat.transforms import (
    apply_transform,
    get_transform_for_param,
    TransformType,
    describe_transform,
)


class TestConv1dWeightTransform:
    """Tests for Conv1d weight transforms: JAX (K, In, Out) -> PyTorch (Out, In, K)."""

    def test_encoder_conv1(self):
        """Test DnaEmbedder first conv weight transform."""
        jax = np.random.randn(15, 4, 768).astype(np.float32)  # (K, In, Out)
        pt_shape = (768, 4, 15)

        result = apply_transform("encoder.dna_embedder.conv1.weight", jax, pt_shape)

        assert result.shape == pt_shape
        # Verify the transform is correct: result[out, in, k] = jax[k, in, out]
        assert np.allclose(result[0, 0, 0], jax[0, 0, 0])
        assert np.allclose(result[100, 2, 10], jax[10, 2, 100])
        assert np.allclose(result, jax.transpose(2, 1, 0))

    def test_encoder_block_conv(self):
        """Test DnaEmbedder block conv weight transform."""
        jax = np.random.randn(5, 768, 768).astype(np.float32)
        pt_shape = (768, 768, 5)

        result = apply_transform("encoder.dna_embedder.block.conv.weight", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result, jax.transpose(2, 1, 0))

    def test_down_block_conv(self):
        """Test DownResBlock conv weight transform."""
        jax = np.random.randn(5, 768, 896).astype(np.float32)
        pt_shape = (896, 768, 5)

        result = apply_transform("encoder.down_blocks.0.block1.conv.weight", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result, jax.transpose(2, 1, 0))

    def test_up_block_conv_in(self):
        """Test UpResBlock conv_in weight transform."""
        jax = np.random.randn(5, 1536, 768).astype(np.float32)
        pt_shape = (768, 1536, 5)

        result = apply_transform("decoder.up_blocks.0.conv_in.conv.weight", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result, jax.transpose(2, 1, 0))

    def test_up_block_conv_out(self):
        """Test UpResBlock conv_out weight transform."""
        jax = np.random.randn(5, 768, 768).astype(np.float32)
        pt_shape = (768, 768, 5)

        result = apply_transform("decoder.up_blocks.0.conv_out.conv.weight", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result, jax.transpose(2, 1, 0))

    def test_conv1d_wrong_ndim_raises(self):
        """Test that Conv1d transform fails with wrong ndim (2D instead of 3D)."""
        jax = np.random.randn(768, 768).astype(np.float32)  # 2D instead of 3D
        pt_shape = (768, 768, 5)

        # The encoder.dna_embedder.conv1 pattern matches CONV1D_WEIGHT
        # but the transform expects 3D and we have 2D
        with pytest.raises(ValueError, match="CONV1D_WEIGHT expects 3D"):
            apply_transform("encoder.dna_embedder.conv1.weight", jax, pt_shape)


class TestLinearWeightTransform:
    """Tests for Linear weight transforms: JAX (In, Out) -> PyTorch (Out, In)."""

    def test_mha_q_proj(self):
        """Test MHA query projection weight transform."""
        jax = np.random.randn(1536, 1024).astype(np.float32)  # (In, Out)
        pt_shape = (1024, 1536)

        result = apply_transform("tower.blocks.0.mha.q_proj.weight", jax, pt_shape)

        assert result.shape == pt_shape
        # Verify ALL values: result[out, in] == jax[in, out]
        assert np.allclose(result, jax.T)

    def test_mha_linear_embedding(self):
        """Test MHA output projection weight transform."""
        jax = np.random.randn(1536, 1536).astype(np.float32)
        pt_shape = (1536, 1536)

        result = apply_transform("tower.blocks.0.mha.linear_embedding.weight", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result, jax.T)

    def test_mlp_fc1(self):
        """Test MLP fc1 weight transform."""
        jax = np.random.randn(1536, 3072).astype(np.float32)
        pt_shape = (3072, 1536)

        result = apply_transform("tower.blocks.0.mlp.fc1.weight", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result, jax.T)

    def test_pair_update_linear_q(self):
        """Test PairUpdate linear_q weight transform."""
        jax = np.random.randn(128, 128).astype(np.float32)
        pt_shape = (128, 128)

        result = apply_transform(
            "tower.blocks.0.pair_update.row_attn.linear_q.weight", jax, pt_shape
        )

        assert result.shape == pt_shape
        assert np.allclose(result, jax.T)

    def test_embedder_project_in(self):
        """Test OutputEmbedder project_in weight transform (now Conv1d k=1)."""
        jax = np.random.randn(1536, 3072).astype(np.float32)
        pt_shape = (3072, 1536, 1)  # NCL format: Conv1d(k=1)

        result = apply_transform("embedder_128bp.project_in.weight", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result[:, :, 0], jax.T)

    def test_linear_wrong_ndim_raises(self):
        """Test that 3D array fails for a linear-expected parameter.

        When a 3D array is passed but the pattern expects 2D (via condition),
        no pattern matches and an error is raised.
        """
        jax = np.random.randn(15, 768, 768).astype(np.float32)  # 3D instead of 2D
        pt_shape = (768, 768)

        # The LINEAR_WEIGHT pattern has a condition: len(s) == 2
        # So 3D array won't match, raising "No matching pattern" error
        with pytest.raises(ValueError, match="No matching pattern"):
            apply_transform("tower.blocks.0.mlp.fc1.weight", jax, pt_shape)


class TestPointwiseEdgeCase:
    """Tests for the pointwise edge case: now uses Conv1d(k=1) in NCL format."""

    def test_pointwise_detected_as_linear_to_conv1d(self):
        """Pointwise ConvBlock (kernel_size=1) now uses Conv1d, should be detected."""
        jax_shape = (1536, 768)

        transform = get_transform_for_param(
            "decoder.up_blocks.0.pointwise.conv.weight", jax_shape
        )

        assert transform == TransformType.LINEAR_TO_CONV1D

    def test_pointwise_transform(self):
        """Test pointwise projection weight transform to Conv1d(k=1) format."""
        jax = np.random.randn(768, 768).astype(np.float32)
        pt_shape = (768, 768, 1)  # NCL format: Conv1d(k=1)

        result = apply_transform("decoder.up_blocks.0.pointwise.conv.weight", jax, pt_shape)

        assert result.shape == pt_shape
        # Should be transposed and expanded
        assert np.allclose(result[:, :, 0], jax.T)


class TestMultiOrganismConv1d:
    """Tests for MultiOrganismConv1d weights: NCL format with transposed dims."""

    def test_heads_atac_multi_org_conv1d(self):
        """Test GenomeTracksHead MultiOrganismConv1d - swap last two dims."""
        jax = np.random.randn(2, 3072, 256).astype(np.float32)  # (NumOrg, In, Out)
        pt_shape = (2, 256, 3072)  # (NumOrg, Out, In)

        result = apply_transform("heads.atac.convs.1.weight", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result, jax.transpose(0, 2, 1))

    def test_contact_maps_head_no_transform(self):
        """Test ContactMapsHead MultiOrganismLinear - same shape (NLC pair activations)."""
        jax = np.random.randn(2, 128, 28).astype(np.float32)
        pt_shape = (2, 128, 28)

        result = apply_transform("contact_maps_head.linear.weight", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result, jax)

    def test_splice_classification_head_multi_org_conv1d(self):
        """Test SpliceClassificationHead MultiOrganismConv1d - swap last two dims."""
        jax = np.random.randn(2, 1536, 5).astype(np.float32)  # (NumOrg, In, Out)
        pt_shape = (2, 5, 1536)  # (NumOrg, Out, In)

        result = apply_transform("splice_sites_classification_head.conv.weight", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result, jax.transpose(0, 2, 1))


class TestOrganismEmbedding:
    """Tests for Organism embedding transforms: JAX (Dim, NumOrg) -> PyTorch (NumOrg, Dim)."""

    def test_main_organism_embed(self):
        """Test main organism embedding transpose."""
        jax = np.random.randn(1536, 2).astype(np.float32)  # (Dim, NumOrg)
        pt_shape = (2, 1536)

        result = apply_transform("organism_embed.weight", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result, jax.T)

    def test_embedder_organism_embed(self):
        """Test output embedder organism embedding transpose."""
        jax = np.random.randn(3072, 2).astype(np.float32)
        pt_shape = (2, 3072)

        result = apply_transform("embedder_128bp.organism_embed.weight", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result, jax.T)


class TestConvScale:
    """Tests for Conv scale transforms: JAX (1, 1, C) -> PyTorch (C, 1, 1)."""

    def test_conv_scale_3d_to_3d(self):
        """Test conv scale transpose from JAX to PyTorch format."""
        jax = np.random.randn(1, 1, 768).astype(np.float32)  # JAX (1, 1, C)
        pt_shape = (768, 1, 1)

        result = apply_transform("encoder.dna_embedder.block.conv.scale", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result[:, 0, 0], jax[0, 0, :])
        assert np.allclose(result, jax.transpose(2, 1, 0))

    def test_conv_scale_1d_to_3d(self):
        """Test conv scale reshape from 1D to 3D."""
        jax = np.random.randn(768).astype(np.float32)  # JAX 1D
        pt_shape = (768, 1, 1)

        result = apply_transform("encoder.down_blocks.0.block1.conv.scale", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result.squeeze(), jax)


class TestNormParameters:
    """Tests for normalization parameter transforms: squeeze to 1D."""

    def test_norm_weight_squeeze_3d(self):
        """Test norm weight squeeze from 3D to 1D."""
        jax = np.random.randn(1, 1, 768).astype(np.float32)
        pt_shape = (768,)

        result = apply_transform("encoder.dna_embedder.block.norm.weight", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result, jax.squeeze())

    def test_norm_bias_squeeze(self):
        """Test norm bias squeeze."""
        jax = np.random.randn(1, 1, 768).astype(np.float32)
        pt_shape = (768,)

        result = apply_transform("tower.blocks.0.mha.norm.bias", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result, jax.squeeze())

    def test_norm_running_var_squeeze(self):
        """Test norm running_var squeeze."""
        jax = np.random.randn(1, 1, 768).astype(np.float32)
        pt_shape = (768,)

        result = apply_transform("tower.blocks.0.mlp.norm.running_var", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result, jax.squeeze())


class TestResidualScale:
    """Tests for residual scale transforms: scalar () -> (1,)."""

    def test_residual_scale_scalar_to_1d(self):
        """Test residual scale reshape from scalar to 1D."""
        jax = np.array(1.0, dtype=np.float32)  # scalar
        pt_shape = (1,)

        result = apply_transform("decoder.up_blocks.0.residual_scale", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result[0], jax)


class TestBiasParameters:
    """Tests for bias parameters: NO transform needed."""

    def test_linear_bias_no_transform(self):
        """Test linear bias - same shape."""
        jax = np.random.randn(1024).astype(np.float32)
        pt_shape = (1024,)

        result = apply_transform("tower.blocks.0.mha.linear_embedding.bias", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result, jax)

    def test_q_r_bias_no_transform(self):
        """Test q_r_bias - same shape (4D)."""
        jax = np.random.randn(1, 1, 32, 128).astype(np.float32)
        pt_shape = (1, 1, 32, 128)

        result = apply_transform(
            "tower.blocks.0.pair_update.seq2pair.q_r_bias", jax, pt_shape
        )

        assert result.shape == pt_shape
        assert np.allclose(result, jax)


class TestShapesAlreadyMatch:
    """Tests for cases where shapes already match."""

    def test_shapes_match_returns_copy(self):
        """When shapes match, should return a copy of the input."""
        jax = np.random.randn(768, 768).astype(np.float32)
        pt_shape = (768, 768)

        result = apply_transform("some.bias", jax, pt_shape)

        assert result.shape == pt_shape
        assert np.allclose(result, jax)
        # Should be a copy, not the same object
        assert result is not jax


class TestGetTransformForParam:
    """Tests for transform type detection."""

    def test_multi_org_conv1d_detected(self):
        """MultiOrganismConv1d should be detected correctly."""
        transform = get_transform_for_param("heads.atac.convs.1.weight", (2, 3072, 256))
        assert transform == TransformType.MULTI_ORG_CONV1D

    def test_conv_detected_correctly(self):
        """Conv weights should be detected correctly."""
        transform = get_transform_for_param(
            "encoder.down_blocks.0.block1.conv.weight", (5, 768, 896)
        )
        assert transform == TransformType.CONV1D_WEIGHT

    def test_linear_detected_correctly(self):
        """Linear weights should be detected correctly."""
        transform = get_transform_for_param(
            "tower.blocks.0.mlp.fc1.weight", (1536, 3072)
        )
        assert transform == TransformType.LINEAR_WEIGHT

    def test_unknown_pattern_raises(self):
        """Unknown parameter patterns should raise ValueError."""
        with pytest.raises(ValueError, match="No matching pattern"):
            get_transform_for_param("some.unknown.param.xyz", (100, 100))


class TestDescribeTransform:
    """Tests for transform description utility."""

    def test_describe_all_types(self):
        """All transform types should have descriptions."""
        for transform_type in TransformType:
            desc = describe_transform(transform_type)
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestTransformRoundtrip:
    """Tests for transform consistency and reversibility."""

    def test_conv1d_transpose_roundtrip(self):
        """Applying Conv1D transform twice with opposite transposes restores original."""
        # JAX (K, In, Out) -> PyTorch (Out, In, K) -> back to (K, In, Out)
        jax = np.random.randn(15, 4, 768).astype(np.float32)
        pt_shape = (768, 4, 15)

        # Forward: JAX -> PyTorch
        pt_result = apply_transform("encoder.dna_embedder.conv1.weight", jax, pt_shape)
        assert pt_result.shape == pt_shape

        # Reverse manually: (Out, In, K) -> (K, In, Out)
        back = np.transpose(pt_result, (2, 1, 0))
        assert np.allclose(back, jax)

    def test_linear_transpose_roundtrip(self):
        """Applying Linear transform twice restores original."""
        jax = np.random.randn(1536, 3072).astype(np.float32)
        pt_shape = (3072, 1536)

        pt_result = apply_transform("tower.blocks.0.mlp.fc1.weight", jax, pt_shape)
        back = pt_result.T
        assert np.allclose(back, jax)

    def test_all_transform_types_produce_correct_output_shape(self):
        """Every transform should produce the exact requested output shape."""
        test_cases = [
            ("encoder.dna_embedder.conv1.weight", (15, 4, 768), (768, 4, 15)),
            ("tower.blocks.0.mlp.fc1.weight", (1536, 3072), (3072, 1536)),
            ("organism_embed.weight", (1536, 2), (2, 1536)),
            ("heads.atac.convs.1.weight", (2, 3072, 256), (2, 256, 3072)),  # NCL format
            ("tower.blocks.0.mha.linear_embedding.bias", (1024,), (1024,)),
            ("embedder_128bp.project_in.weight", (1536, 3072), (3072, 1536, 1)),  # LINEAR_TO_CONV1D
            ("decoder.up_blocks.0.pointwise.conv.weight", (768, 768), (768, 768, 1)),  # LINEAR_TO_CONV1D
        ]

        for param_name, jax_shape, pt_shape in test_cases:
            jax = np.random.randn(*jax_shape).astype(np.float32)
            result = apply_transform(param_name, jax, pt_shape)
            assert result.shape == pt_shape, (
                f"Shape mismatch for {param_name}: expected {pt_shape}, got {result.shape}"
            )

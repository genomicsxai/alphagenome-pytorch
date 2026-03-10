"""Unit tests for head modules (no JAX dependency)."""

import pytest
import torch


@pytest.mark.unit
class TestMultiOrganismConv1d:
    """Tests for MultiOrganismConv1d layer (NCL format)."""

    def test_initialization(self):
        """Test layer initializes with correct parameters."""
        from alphagenome_pytorch.heads import MultiOrganismConv1d

        layer = MultiOrganismConv1d(in_channels=64, out_channels=32, num_organisms=2)

        assert layer.in_channels == 64
        assert layer.out_channels == 32
        assert layer.num_organisms == 2
        # NCL format: weight is (num_org, out_channels, in_channels)
        assert layer.weight.shape == (2, 32, 64)
        assert layer.bias.shape == (2, 32)

    def test_forward_shape(self):
        """Test output shape is correct (NCL format)."""
        from alphagenome_pytorch.heads import MultiOrganismConv1d

        layer = MultiOrganismConv1d(in_channels=64, out_channels=32, num_organisms=2)

        B, C, S = 4, 64, 100  # NCL format
        x = torch.randn(B, C, S)
        org_idx = torch.tensor([0, 0, 1, 1])

        output = layer(x, org_idx)
        assert output.shape == (B, 32, S)  # NCL format

    def test_organism_specific_weights(self):
        """Test that different organisms use different weights."""
        from alphagenome_pytorch.heads import MultiOrganismConv1d

        layer = MultiOrganismConv1d(in_channels=64, out_channels=32, num_organisms=2)

        # Make weights very different for each organism
        layer.weight.data[0] = torch.ones_like(layer.weight.data[0])
        layer.weight.data[1] = torch.zeros_like(layer.weight.data[1])
        layer.bias.data.fill_(0)

        B, C, S = 2, 64, 10  # NCL format
        x = torch.randn(B, C, S)

        # Human organism (index 0)
        out_human = layer(x, torch.tensor([0, 0]))
        # Mouse organism (index 1)
        out_mouse = layer(x, torch.tensor([1, 1]))

        # Outputs should be very different
        assert not torch.allclose(out_human, out_mouse)


@pytest.mark.unit
class TestGenomeTracksHead:
    """Tests for GenomeTracksHead class. Outputs NLC format (B, S, T)."""

    def test_initialization(self):
        """Test head initializes with correct parameters."""
        from alphagenome_pytorch.heads import GenomeTracksHead

        head = GenomeTracksHead(
            in_channels=None,
            num_tracks=256,
            resolutions=[1, 128],
            num_organisms=2,
            apply_squashing=False,
        )

        assert head.num_tracks == 256
        assert head.resolutions == [1, 128]
        assert "1" in head.convs  # Conv1d on NCL, equivalent to JAX Linear on NLC
        assert "128" in head.convs
        assert "1" in head.residual_scales
        assert "128" in head.residual_scales

    def test_initialization_with_track_means(self):
        """Test head initializes with provided track means."""
        from alphagenome_pytorch.heads import GenomeTracksHead

        track_means = torch.rand(2, 256) * 0.5 + 0.1
        head = GenomeTracksHead(
            in_channels=None,
            num_tracks=256,
            resolutions=[1, 128],
            num_organisms=2,
            track_means=track_means,
        )

        assert torch.allclose(head.track_means, track_means)

    def test_forward_shapes(self):
        """Test head produces correct output shapes (NLC format)."""
        from alphagenome_pytorch.heads import GenomeTracksHead

        head = GenomeTracksHead(
            in_channels=None,
            num_tracks=256,
            resolutions=[1, 128],
            num_organisms=2,
        )

        B, S_1bp, S_128bp = 2, 131072, 1024
        # Input NCL format: (B, C, S)
        embeddings = {
            1: torch.randn(B, 1536, S_1bp),
            128: torch.randn(B, 3072, S_128bp),
        }
        organism_index = torch.tensor([0, 1])

        outputs = head(embeddings, organism_index)

        # Output NLC format: (B, S, T) where T is num_tracks
        assert outputs[1].shape == (B, S_1bp, 256)
        assert outputs[128].shape == (B, S_128bp, 256)

    def test_single_resolution(self):
        """Test head with single resolution (like ChIP-seq)."""
        from alphagenome_pytorch.heads import GenomeTracksHead

        head = GenomeTracksHead(
            in_channels=None,
            num_tracks=1664,
            resolutions=[128],  # ChIP-seq style
            num_organisms=2,
        )

        assert head.resolutions == [128]
        assert "128" in head.convs  # NCL uses convs not linears
        assert "1" not in head.convs

    def test_output_non_negative(self):
        """Test that outputs are non-negative after softplus."""
        from alphagenome_pytorch.heads import GenomeTracksHead

        head = GenomeTracksHead(
            in_channels=None,
            num_tracks=256,
            resolutions=[128],
            num_organisms=2,
        )

        B, S = 2, 1024
        # Input NCL format: (B, C, S)
        embeddings = {128: torch.randn(B, 3072, S)}
        organism_index = torch.tensor([0, 1])

        outputs = head(embeddings, organism_index)
        # Output NLC format: (B, S, T)
        assert (outputs[128] >= 0).all()

    def test_skips_missing_resolutions(self):
        """Head should skip resolutions not present in embeddings_dict.

        When model.forward() is called with resolutions=(128,), only 128bp
        embeddings are provided. Heads built with [1, 128] should gracefully
        skip the missing 1bp resolution instead of raising KeyError.
        """
        from alphagenome_pytorch.heads import GenomeTracksHead

        head = GenomeTracksHead(
            in_channels=None,
            num_tracks=256,
            resolutions=[1, 128],
            num_organisms=2,
        )

        B, S_128bp = 2, 1024
        # Only provide 128bp embeddings (simulating resolutions=(128,))
        # Input NCL format: (B, C, S)
        embeddings = {128: torch.randn(B, 3072, S_128bp)}
        organism_index = torch.tensor([0, 1])

        outputs = head(embeddings, organism_index)

        assert 128 in outputs
        assert 1 not in outputs
        assert outputs[128].shape == (B, S_128bp, 256)


@pytest.mark.unit
class TestContactMapsHead:
    """Tests for ContactMapsHead class."""

    def test_initialization(self):
        """Test contact head initializes correctly."""
        from alphagenome_pytorch.heads import ContactMapsHead

        head = ContactMapsHead(
            in_features=128,
            num_tracks=28,
            num_organisms=2,
        )

        assert head.num_tracks == 28
        assert head.num_organisms == 2

    def test_forward_shape(self):
        """Test contact head forward pass shape."""
        from alphagenome_pytorch.heads import ContactMapsHead

        head = ContactMapsHead(in_features=128, num_tracks=28, num_organisms=2)

        B, S = 2, 1024
        pair_embeddings = torch.randn(B, S, S, 128)
        organism_index = torch.tensor([0, 1])

        output = head(pair_embeddings, organism_index)

        assert output.shape == (B, S, S, 28)

    def test_small_input(self):
        """Test with smaller input for quick verification."""
        from alphagenome_pytorch.heads import ContactMapsHead

        head = ContactMapsHead(in_features=128, num_tracks=28, num_organisms=2)

        B, S = 1, 64
        pair_embeddings = torch.randn(B, S, S, 128)
        organism_index = torch.tensor([0])

        output = head(pair_embeddings, organism_index)

        assert output.shape == (B, S, S, 28)


@pytest.mark.unit
class TestPredictionsScaling:
    """Tests for predictions_scaling function (NLC format)."""

    def test_basic_scaling(self):
        """Test predictions_scaling function (NLC format)."""
        from alphagenome_pytorch.heads import predictions_scaling

        B, S, C = 2, 100, 10  # NLC format
        x = torch.randn(B, S, C)
        track_means = torch.ones(B, C) * 0.5

        scaled = predictions_scaling(
            x, track_means, resolution=128, apply_squashing=False
        )

        assert scaled.shape == x.shape

    def test_squashing(self):
        """Test predictions_scaling with squashing (RNA-seq mode)."""
        from alphagenome_pytorch.heads import predictions_scaling

        B, S, C = 2, 100, 10  # NLC format
        x = torch.abs(torch.randn(B, S, C)) + 0.1  # Positive values
        track_means = torch.ones(B, C) * 0.5

        scaled_no_squash = predictions_scaling(
            x.clone(), track_means, resolution=128, apply_squashing=False
        )
        scaled_with_squash = predictions_scaling(
            x.clone(), track_means, resolution=128, apply_squashing=True
        )

        # With squashing (power 1/0.75), values should be larger
        # for x > 1 after soft clipping
        assert scaled_with_squash.shape == scaled_no_squash.shape

    def test_resolution_scaling(self):
        """Test that resolution affects scaling."""
        from alphagenome_pytorch.heads import predictions_scaling

        B, S, C = 2, 100, 10  # NLC format
        x = torch.abs(torch.randn(B, S, C))
        track_means = torch.ones(B, C)

        scaled_1bp = predictions_scaling(
            x.clone(), track_means, resolution=1, apply_squashing=False
        )
        scaled_128bp = predictions_scaling(
            x.clone(), track_means, resolution=128, apply_squashing=False
        )

        # 128bp should be scaled 128x more
        # (after soft clipping which may affect this)
        assert not torch.allclose(scaled_1bp, scaled_128bp)

@pytest.mark.unit
class TestSpliceSitesClassificationHead:
    def test_forward_shape(self):
        """Test forward pass shape (NLC output)."""
        from alphagenome_pytorch.heads import SpliceSitesClassificationHead

        head = SpliceSitesClassificationHead(in_channels=128)

        B, C, S = 2, 128, 1024  # Input NCL format
        embeddings = torch.randn(B, C, S)
        organism_index = torch.tensor([0, 1])

        output = head(embeddings, organism_index)
        # Output is NLC: (B, S, 5)
        assert output['logits'].shape == (B, S, 5)
        assert output['probs'].shape == (B, S, 5)

        # Test that we have probabilities (sum over channel dim=-1)
        assert (output['probs'] >= 0).all()
        assert torch.allclose(output['probs'].sum(dim=-1), torch.ones(B, S), atol=1e-6)

@pytest.mark.unit
class TestSpliceSitesUsageHead:
    def test_forward_shape(self):
        """Test forward pass shape (NLC output)."""
        from alphagenome_pytorch.heads import SpliceSitesUsageHead

        num_tissues_x_strands = 28

        head = SpliceSitesUsageHead(in_channels=128, num_output_tracks=num_tissues_x_strands)

        B, C, S = 2, 128, 1024  # Input NCL format
        embeddings = torch.randn(B, C, S)
        organism_index = torch.tensor([0, 1])

        output = head(embeddings, organism_index)
        # Output is NLC: (B, S, T)
        assert output['logits'].shape == (B, S, num_tissues_x_strands)
        assert output['predictions'].shape == (B, S, num_tissues_x_strands)

        # Test that we have sigmoid values
        assert (output['predictions'] >= 0).all()
        assert (output['predictions'] <= 1).all()

    @pytest.mark.parametrize(
        ("num_organisms", "num_tracks_per_organism"),
        [
            (1, (734,)),
            (2, (734, 180)),
        ],
    )
    def test_track_mask_for_one_and_two_organisms(
        self,
        num_organisms,
        num_tracks_per_organism,
    ):
        """Mask shape/counts should match per-organism track counts."""
        from alphagenome_pytorch.heads import SpliceSitesUsageHead

        head = SpliceSitesUsageHead(
            in_channels=128,
            num_output_tracks=734,
            num_organisms=num_organisms,
            num_tracks_per_organism=num_tracks_per_organism,
        )

        assert head.track_mask.shape == (num_organisms, 734)
        for org_idx, expected_tracks in enumerate(num_tracks_per_organism):
            assert int(head.track_mask[org_idx].sum().item()) == expected_tracks

    def test_forward_returns_jax_style_track_mask(self):
        """Forward should return a per-batch track mask usable in loss."""
        from alphagenome_pytorch.heads import SpliceSitesUsageHead

        head = SpliceSitesUsageHead(
            in_channels=128,
            num_output_tracks=734,
            num_organisms=2,
            num_tracks_per_organism=(734, 180),
        )

        B, C, S = 2, 128, 64
        embeddings = torch.randn(B, C, S)
        organism_index = torch.tensor([0, 1])
        output = head(embeddings, organism_index, channels_last=True)

        track_mask = output['track_mask']
        assert track_mask.dtype == torch.bool
        assert track_mask.shape == (B, 1, 734)
        assert bool(track_mask[0].all())
        assert bool(track_mask[1, :, :180].all())
        assert not bool(track_mask[1, :, 180:].any())

@pytest.mark.unit
class TestSpliceSitesJunctionHead:
    def test_forward_shape(self):
        """Test forward pass shape (NCL format)."""
        from alphagenome_pytorch.heads import SpliceSitesJunctionHead
        B, S, P, T = 2, 1024, 10, 367  # Use full tissue range to test padding
        in_channels = 128
        hidden_dim = 64
        head = SpliceSitesJunctionHead(in_channels=in_channels, hidden_dim=hidden_dim, num_tissues=T)
        # NCL format: (B, C, S)
        embeddings = torch.randn(B, in_channels, S)
        organism_index = torch.tensor([0, 1])

        output = head(embeddings, organism_index, splice_site_positions=torch.randint(0, S, (B, 4, P)))
        assert output['pred_counts'].shape == (B, P, P, 2 * T)

    def test_no_splice_site_positions(self):
        """Test that missing splice_site_positions raises error."""
        from alphagenome_pytorch.heads import SpliceSitesJunctionHead
        B, S, P, T = 2, 1024, 10, 367  # Use full tissue range to test padding
        in_channels = 128
        hidden_dim = 64

        head = SpliceSitesJunctionHead(in_channels=in_channels, hidden_dim=hidden_dim, num_tissues=T)
        # NCL format: (B, C, S)
        embeddings = torch.randn(B, in_channels, S)
        organism_index = torch.tensor([0, 1])

        with pytest.raises(ValueError):
            output = head(embeddings, organism_index)

    @pytest.mark.parametrize(
        ("num_organisms", "num_tracks_per_organism"),
        [
            (1, (367,)),
            (2, (367, 90)),
        ],
    )
    def test_track_mask_for_one_and_two_organisms(
        self,
        num_organisms,
        num_tracks_per_organism,
    ):
        """Tests for one/two-organism junction masks."""
        from alphagenome_pytorch.heads import SpliceSitesJunctionHead

        T = 367
        head = SpliceSitesJunctionHead(
            in_channels=128,
            hidden_dim=64,
            num_tissues=T,
            num_organisms=num_organisms,
            num_tracks_per_organism=num_tracks_per_organism,
        )

        assert head.tissue_mask.shape == (num_organisms, T)
        for org_idx, expected_tissues in enumerate(num_tracks_per_organism):
            assert int(head.tissue_mask[org_idx].sum().item()) == expected_tissues

    def test_forward_applies_per_organism_mask(self):
        """Mouse channels should be masked beyond its configured tissue count."""
        from alphagenome_pytorch.heads import SpliceSitesJunctionHead

        B, S, P, T = 2, 512, 8, 367
        head = SpliceSitesJunctionHead(
            in_channels=128,
            hidden_dim=64,
            num_tissues=T,
            num_organisms=2,
            num_tracks_per_organism=(367, 90),
        )
        embeddings = torch.randn(B, 128, S)
        organism_index = torch.tensor([0, 1])
        positions = torch.randint(0, S, (B, 4, P))

        output = head(embeddings, organism_index, splice_site_positions=positions)
        mask = output['splice_junction_mask']

        # Organism 0 (human): all tissues valid in both strands.
        assert bool(mask[0].all())

        # Organism 1 (mouse): channels >=90 are masked in both concatenated halves.
        assert not bool(mask[1, :, :, 90:T].any())
        assert not bool(mask[1, :, :, T + 90:2 * T].any())


@pytest.mark.unit
class TestScalingFunctions:
    """Tests for predictions_scaling and targets_scaling functions (NLC format)."""

    def test_targets_scaling_reversibility_no_squashing(self):
        """Test that predictions_scaling and targets_scaling are inverses (no squashing)."""
        from alphagenome_pytorch.heads import predictions_scaling, targets_scaling

        batch_size = 2
        num_tracks = 256
        seq_len = 1024

        # Test data with positive values and wide range - NLC format (B, S, C)
        x = torch.randn(batch_size, seq_len, num_tracks).abs() * 50
        track_means = torch.ones(batch_size, num_tracks)
        resolution = 128
        apply_squashing = False

        # Forward: experimental → model → experimental
        scaled = targets_scaling(x, track_means, resolution, apply_squashing)
        unscaled = predictions_scaling(scaled, track_means, resolution, apply_squashing)

        # Should be identity
        torch.testing.assert_close(x, unscaled, rtol=1e-5, atol=1e-6)

        # Reverse: model → experimental → model
        unscaled2 = predictions_scaling(x, track_means, resolution, apply_squashing)
        scaled2 = targets_scaling(unscaled2, track_means, resolution, apply_squashing)

        torch.testing.assert_close(x, scaled2, rtol=1e-5, atol=1e-6)

    def test_targets_scaling_reversibility_with_squashing(self):
        """Test that predictions_scaling and targets_scaling are inverses (with squashing)."""
        from alphagenome_pytorch.heads import predictions_scaling, targets_scaling

        batch_size = 2
        num_tracks = 256
        seq_len = 1024

        # Test data with positive values and wide range - NLC format (B, S, C)
        x = torch.randn(batch_size, seq_len, num_tracks).abs() * 50
        track_means = torch.ones(batch_size, num_tracks)
        resolution = 128
        apply_squashing = True  # RNA-seq mode

        # Forward: experimental → model → experimental
        scaled = targets_scaling(x, track_means, resolution, apply_squashing)
        unscaled = predictions_scaling(scaled, track_means, resolution, apply_squashing)

        # Should be identity
        torch.testing.assert_close(x, unscaled, rtol=1e-5, atol=1e-6)

        # Reverse: model → experimental → model
        unscaled2 = predictions_scaling(x, track_means, resolution, apply_squashing)
        scaled2 = targets_scaling(unscaled2, track_means, resolution, apply_squashing)

        torch.testing.assert_close(x, scaled2, rtol=1e-5, atol=1e-6)

    def test_scaling_with_nonuniform_track_means(self):
        """Test scaling with non-uniform track means."""
        from alphagenome_pytorch.heads import predictions_scaling, targets_scaling

        batch_size = 2
        num_tracks = 128
        seq_len = 512

        # Non-uniform track means (different per organism and track) - NLC format (B, S, C)
        x = torch.randn(batch_size, seq_len, num_tracks).abs() * 20
        track_means = torch.rand(batch_size, num_tracks) * 10 + 0.1  # Avoid zeros
        resolution = 1

        for apply_squashing in [False, True]:
            # Forward cycle
            scaled = targets_scaling(x, track_means, resolution, apply_squashing)
            unscaled = predictions_scaling(scaled, track_means, resolution, apply_squashing)
            torch.testing.assert_close(x, unscaled, rtol=1e-5, atol=1e-6)

            # Reverse cycle
            unscaled2 = predictions_scaling(x, track_means, resolution, apply_squashing)
            scaled2 = targets_scaling(unscaled2, track_means, resolution, apply_squashing)
            torch.testing.assert_close(x, scaled2, rtol=1e-5, atol=1e-6)

    def test_scaling_soft_clipping_region(self):
        """Test that soft clipping region is properly reversed."""
        from alphagenome_pytorch.heads import predictions_scaling, targets_scaling

        batch_size = 1
        num_tracks = 10
        seq_len = 100

        # Create data with values both below and above soft_clip_value (10.0) - NLC format (B, S, C)
        x = torch.cat([
            torch.linspace(0, 5, 50).unsqueeze(0).unsqueeze(-1).expand(1, 50, num_tracks),
            torch.linspace(10, 50, 50).unsqueeze(0).unsqueeze(-1).expand(1, 50, num_tracks),
        ], dim=1)

        track_means = torch.ones(batch_size, num_tracks)
        resolution = 128

        # Test both modes
        for apply_squashing in [False, True]:
            scaled = targets_scaling(x, track_means, resolution, apply_squashing)
            unscaled = predictions_scaling(scaled, track_means, resolution, apply_squashing)
            torch.testing.assert_close(x, unscaled, rtol=1e-4, atol=1e-5)

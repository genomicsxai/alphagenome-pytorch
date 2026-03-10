"""
Integration tests for fine-tuning heads with full model embeddings.

Tests that finetuning heads work correctly with real model outputs.
"""

import gc

import pytest
import torch

from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head


@pytest.mark.integration
class TestHeadWithEmbeddings:
    """Tests for heads with real model embedding outputs."""

    @pytest.fixture(scope="class")
    def model_fp32(self):
        """Shared full_float32 model instance."""
        from alphagenome_pytorch import AlphaGenome
        from alphagenome_pytorch.config import DtypePolicy

        model = AlphaGenome(dtype_policy=DtypePolicy.full_float32())
        yield model
        del model
        gc.collect()

    @pytest.fixture(scope="class")
    def model_mixed(self):
        """Shared mixed_precision model instance."""
        from alphagenome_pytorch import AlphaGenome
        from alphagenome_pytorch.config import DtypePolicy

        model = AlphaGenome(dtype_policy=DtypePolicy.mixed_precision())
        yield model
        del model
        gc.collect()

    @pytest.fixture(params=["full_float32", "mixed_precision"])
    def model_and_autocast(self, request, model_fp32, model_mixed):
        """Parametrized fixture providing model and autocast settings."""
        if request.param == "full_float32":
            return model_fp32, False
        return model_mixed, True

    def test_rnaseq_head_with_model_embeddings(self, model_and_autocast):
        """Test RNA-seq head works with model embedding shapes."""
        model, use_amp = model_and_autocast
        x = torch.randn(1, 2048, 4)  # Min length given downsampling
        organism_idx = torch.tensor([0])

        # autocast handles weight casting (bf16 input x f32 weights) — matches production usage
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(x, organism_idx, return_embeddings=True, channels_last=False)

            # Create head and run forward
            head = create_finetuning_head('rna_seq', n_tracks=3, resolutions=(128,))
            embeddings_dict = {128: outputs["embeddings_128bp"]}

            preds = head(embeddings_dict, organism_idx)

        assert preds[128].shape == (1, 16, 3)  # Post downsampling at 128bp resolution
        assert (preds[128] >= 0).all()

    def test_atac_head_with_model_embeddings(self, model_and_autocast):
        """Test ATAC head works with model embedding shapes."""
        model, use_amp = model_and_autocast
        x = torch.randn(1, 2048, 4)  # Min length given downsampling
        organism_idx = torch.tensor([0])

        # autocast handles weight casting (bf16 input x f32 weights) — matches production usage
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(x, organism_idx, return_embeddings=True, channels_last=False)

            # Create head and run forward
            head = create_finetuning_head('atac', n_tracks=5, resolutions=(128,))
            embeddings_dict = {128: outputs["embeddings_128bp"]}

            preds = head(embeddings_dict, organism_idx)

        assert preds[128].shape == (1, 16, 5)  # Post downsampling at 128bp resolution
        assert (preds[128] >= 0).all()

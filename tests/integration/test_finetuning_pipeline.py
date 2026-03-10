"""Integration tests for fine-tuning pipeline with mock data.

These tests verify that the fine-tuning pipeline works end-to-end:
1. Loading pretrained weights
2. Creating datasets from mock data
3. Training for 1 epoch
4. Saving checkpoints

All tests require --torch-weights flag to load pretrained weights.

Note: These tests use the 'finetuning' marker instead of 'integration' to avoid
requiring JAX checkpoint (which is only needed for JAX comparison tests).

For unit tests (dataset loading, head creation), see tests/unit/test_finetuning_*.py
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.extensions.finetuning.transfer import (
    load_trunk,
    count_trainable_params,
    prepare_for_transfer,
    TransferConfig,
    remove_all_heads,
)
from alphagenome_pytorch.extensions.finetuning.datasets import RNASeqDataset, ATACDataset
from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head
from alphagenome_pytorch.extensions.finetuning.training import (
    MODALITY_CONFIGS,
    collate_genomic,
    train_epoch,
    save_checkpoint,
    create_lr_scheduler,
)
from alphagenome_pytorch.extensions.finetuning.adapters import get_adapter_params, LoRA


@pytest.mark.finetuning
class TestFinetuningPipeline:
    """Integration tests for fine-tuning pipeline with mock data.

    These tests require --torch-weights to load pretrained weights.
    """

    @pytest.fixture
    def finetuning_model(self, torch_weights_path):
        """Load AlphaGenome model from pretrained weights."""
        model = AlphaGenome()
        model = load_trunk(model, str(torch_weights_path), exclude_heads=True)
        return model

    @pytest.mark.parametrize("modality", ["rna_seq", "atac"])
    @pytest.mark.parametrize("sequence_length", [16384, 32768])
    def test_finetuning_pipeline(self, mock_data_dir, finetuning_model, tmp_path, modality, sequence_length):
        """Test fine-tuning pipeline runs for 1 epoch successfully."""
        # Setup
        model = finetuning_model
        model = remove_all_heads(model)

        modality_config = MODALITY_CONFIGS[modality]
        resolutions = modality_config.resolutions
        n_tracks = 2

        # Create TransferConfig
        config = TransferConfig(
            mode="lora",
            lora_targets=["q_proj", "v_proj"],
            lora_rank=8,
            lora_alpha=16,
        )

        # Create head
        head = create_finetuning_head(
            assay_type=modality,
            n_tracks=n_tracks,
            resolutions=resolutions,
        )
        model.heads[modality] = head

        # Prepare for transfer
        model = prepare_for_transfer(model, config)

        # Check that the model has correctly replaced target modules
        # with LoRA modules
        for name, module in model.named_modules():
            if name.endswith("q_proj") or name.endswith("v_proj"):
                assert isinstance(module, LoRA)
        
        # Check that the model has the correct number of trainable parameters
        # With LoRA + unfreeze_norm=False (default), trainable params include:
        # - LoRA adapters
        # - New heads
        # Norm layers are frozen by default (set unfreeze_norm=True to train them)
        trainable_params = count_trainable_params(model)
        print("Num trainable parameters: ", trainable_params)
        expected = trainable_params["adapters"] + trainable_params["heads"]
        assert trainable_params["total"] == expected, f"Expected {expected}, got {trainable_params['total']}"
        assert trainable_params["norm"] == 0, "Norm layers should be frozen by default"
        assert trainable_params["other"] == 0, "Unexpected trainable parameters"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Create dataset with shorter sequence length for testing (saves memory)
        if modality == "rna_seq":
            train_dataset = RNASeqDataset(
                genome_fasta=str(mock_data_dir / "mock_genome.fa"),
                bigwig_files=[str(mock_data_dir / f"mock_rnaseq_track{i}.bw") for i in [1, 2]],
                bed_file=str(mock_data_dir / "mock_positions.bed"),
                resolutions=resolutions,
                sequence_length=sequence_length,
            )
        else:
            train_dataset = ATACDataset(
                genome_fasta=str(mock_data_dir / "mock_genome.fa"),
                bigwig_files=[str(mock_data_dir / f"mock_atac_track{i}.bw") for i in [1, 2]],
                bed_file=str(mock_data_dir / "mock_positions.bed"),
                resolutions=resolutions,
                sequence_length=sequence_length,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_genomic,
        )

        # Setup optimizer
        trainable_params = get_adapter_params(model) + list(head.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
        scheduler = create_lr_scheduler(optimizer, warmup_steps=0, total_steps=len(train_loader))

        # Train 1 epoch
        train_loss = train_epoch(
            model=model,
            head=head,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            resolution_weights=modality_config.default_resolution_weights,
            positional_weight=5.0,
            epoch=1,
            log_every=1,
        )

        # Verify loss is finite
        assert torch.isfinite(torch.tensor(train_loss)), f"Loss is not finite: {train_loss}"

        # Test checkpoint saving
        checkpoint_path = tmp_path / f"test_checkpoint_{modality}.pth"
        save_checkpoint(
            path=checkpoint_path,
            epoch=1,
            model=model,
            optimizer=optimizer,
            val_loss=train_loss,
            track_names=["track1", "track2"],
            modality=modality,
            resolutions=resolutions,
        )
        assert checkpoint_path.exists(), f"Checkpoint not saved: {checkpoint_path}"

        # Verify checkpoint contents
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert "epoch" in checkpoint
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["modality"] == modality
        # Head state is included in model_state_dict (heads are part of the model)
        assert any(k.startswith(f"heads.{modality}") for k in checkpoint["model_state_dict"])
    

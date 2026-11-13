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
    @pytest.mark.parametrize("organism", [0, 1], ids=["human", "mouse"])
    def test_finetuning_pipeline(self, mock_data_dir, finetuning_model, tmp_path, modality, sequence_length, organism):
        """Test fine-tuning pipeline runs for 1 epoch successfully.

        Parametrized over organism so both a human (index 0) and a mouse
        (index 1) fine-tune run without errors: the trunk forwards at the given
        organism (selecting its organism embedding) while the single-organism
        head uses its one slot.
        """
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
            organism=organism,
        )

        # Verify loss is finite (the org-1/mouse forward must not error or NaN)
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
    

    @pytest.mark.parametrize("organism", [0, 1], ids=["human", "mouse"])
    def test_finetune_step_runs_for_both_organisms_no_weights(self, mock_data_dir, organism):
        """A fine-tune epoch runs without errors at organism 0 (human) and 1
        (mouse), from a randomly-initialised model (no pretrained weights).

        This guards the organism fix end-to-end: the trunk forwards at the given
        organism (selecting its organism embedding), the single-organism head
        uses its one slot, and the loss is finite. Mouse (index 1) used to be
        impossible — the trainer hardcoded organism 0.
        """
        torch.manual_seed(0)
        modality = "atac"
        cfg = MODALITY_CONFIGS[modality]

        model = AlphaGenome()
        model = remove_all_heads(model)
        head = create_finetuning_head(assay_type=modality, n_tracks=2, resolutions=cfg.resolutions)
        model.heads[modality] = head
        assert head.num_organisms == 1  # organism-agnostic head

        # Linear-probe: freeze trunk, train the head only.
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True

        device = torch.device("cpu")
        model = model.to(device)

        dataset = ATACDataset(
            genome_fasta=str(mock_data_dir / "mock_genome.fa"),
            bigwig_files=[str(mock_data_dir / f"mock_atac_track{i}.bw") for i in [1, 2]],
            bed_file=str(mock_data_dir / "mock_positions.bed"),
            resolutions=cfg.resolutions,
            sequence_length=16384,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_genomic)
        optimizer = torch.optim.AdamW(list(head.parameters()), lr=1e-4)
        scheduler = create_lr_scheduler(optimizer, warmup_steps=0, total_steps=len(loader))

        train_loss = train_epoch(
            model=model,
            head=head,
            train_loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            resolution_weights=cfg.default_resolution_weights,
            positional_weight=5.0,
            epoch=1,
            log_every=1,
            organism=organism,
        )
        assert torch.isfinite(torch.tensor(train_loss)), f"Loss not finite: {train_loss}"

    def test_organism_index_selects_distinct_embeddings(self):
        """The organism index must actually drive the trunk embedding.

        Forwarding the *same* sequence at organism 0 (human) vs 1 (mouse) must
        produce *different* embeddings (the organism_embed + per-organism output
        embedders are selected by the index), while the same organism is
        deterministic. This is what makes a mouse fine-tune use mouse
        representations rather than silently reusing the human ones.
        """
        torch.manual_seed(0)
        model = AlphaGenome().eval()
        x = torch.randn(1, 16384, 4)
        with torch.no_grad():
            human = model(x, torch.tensor([0]), return_embeddings=True)
            mouse = model(x, torch.tensor([1]), return_embeddings=True)
            human2 = model(x, torch.tensor([0]), return_embeddings=True)
        for key in ("embeddings_128bp", "embeddings_1bp"):
            assert torch.allclose(human[key], human2[key]), (
                f"{key}: same organism (0) should be deterministic"
            )
            assert not torch.allclose(human[key], mouse[key]), (
                f"{key}: human (0) and mouse (1) forwards must differ — the "
                "organism index is not selecting the organism embedding"
            )

    def test_real_organism_embeddings_are_distinct(self, finetuning_model):
        """With real pretrained weights, the trained human and mouse organism
        embeddings are distinct and a mouse (index 1) forward uses the mouse
        representation — not the human one. Requires --torch-weights.
        """
        model = finetuning_model.eval()

        # Trained organism_embed slots must be genuinely different vectors.
        oe = model.organism_embed.weight  # (2, dim): row 0 = human, 1 = mouse
        assert not torch.allclose(oe[0], oe[1])

        torch.manual_seed(0)
        x = torch.randn(1, 16384, 4)
        with torch.no_grad():
            human = model(x, torch.tensor([0]), return_embeddings=True)
            mouse = model(x, torch.tensor([1]), return_embeddings=True)
        for key in ("embeddings_128bp", "embeddings_1bp"):
            assert not torch.allclose(human[key], mouse[key]), (
                f"{key}: human and mouse forwards must differ with real weights"
            )

"""Unit tests for the Trainer API."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from alphagenome_pytorch.extensions.finetuning import Trainer, TrainerConfig, TransferConfig
from alphagenome_pytorch.extensions.finetuning.training import collate_genomic


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv1d(4, 8, kernel_size=1)

    def forward(
        self,
        sequences: torch.Tensor,
        organism_idx: torch.Tensor,
        return_embeddings: bool = True,
        resolutions: tuple[int, ...] = (1, 128),
        channels_last: bool = False,
        encoder_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        del organism_idx, return_embeddings, channels_last
        x = sequences.transpose(1, 2)  # (B, 4, S)
        emb = torch.relu(self.proj(x))  # (B, 8, S)

        if encoder_only:
            return {"encoder_output": emb.transpose(1, 2)}  # (B, S, 8)

        outputs: dict[str, torch.Tensor] = {}
        if 1 in resolutions:
            outputs["embeddings_1bp"] = emb
        if 128 in resolutions:
            outputs["embeddings_128bp"] = emb[:, :, ::2]
        return outputs


class _TinyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resolutions = [1, 128]
        self.num_tracks = 3
        self.convs = nn.ModuleDict({
            "1": nn.Conv1d(8, self.num_tracks, kernel_size=1),
            "128": nn.Conv1d(8, self.num_tracks, kernel_size=1),
        })

    def scale(
        self,
        x: torch.Tensor,
        organism_idx: torch.Tensor,
        resolution: int,
        channels_last: bool = True,
    ) -> torch.Tensor:
        del organism_idx, resolution, channels_last
        return x

    def forward(
        self,
        embeddings_dict: dict[int, torch.Tensor],
        organism_idx: torch.Tensor,
        return_scaled: bool = True,
        channels_last: bool = True,
    ) -> dict[int, torch.Tensor]:
        del organism_idx, return_scaled
        out: dict[int, torch.Tensor] = {}
        for res in self.resolutions:
            if res not in embeddings_dict:
                continue
            pred = torch.nn.functional.softplus(self.convs[str(res)](embeddings_dict[res]))  # (B, T, S)
            if channels_last:
                pred = pred.transpose(1, 2)  # (B, S, T)
            out[res] = pred
        return out


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        del idx
        seq = torch.randn(16, 4)
        targets = {
            1: torch.rand(16, 3),
            128: torch.rand(8, 3),
        }
        return seq, targets


def _make_trainer() -> Trainer:
    model = _TinyBackbone()
    head = _TinyHead()
    params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer_config = TrainerConfig(
        positional_weight=5.0,
        count_weight=1.0,
        max_grad_norm=1.0,
        num_segments=2,
        use_amp=False,
        log_every=1,
    )
    transfer_config = TransferConfig(mode="linear")
    return Trainer(
        model=model,
        heads={"atac": head},
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_config=trainer_config,
        transfer_config=transfer_config,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
    )


@pytest.mark.unit
def test_forward_backward_and_optim_step():
    trainer = _make_trainer()
    batch = next(iter(DataLoader(_TinyDataset(), batch_size=2, collate_fn=collate_genomic)))

    losses = trainer.forward_backward(
        batch=batch,
        modality_weights={"atac": 1.0},
        resolution_weights={"atac": {1: 1.0, 128: 1.0}},
    )
    assert "loss" in losses
    assert losses["loss"] > 0

    step_metrics = trainer.optim_step()
    assert "learning_rate" in step_metrics
    assert step_metrics["learning_rate"] > 0


@pytest.mark.unit
def test_train_epoch_and_validate():
    trainer = _make_trainer()
    train_loader = DataLoader(_TinyDataset(), batch_size=2, collate_fn=collate_genomic)
    val_loader = DataLoader(_TinyDataset(), batch_size=2, collate_fn=collate_genomic)

    train_loss, per_modality = trainer.train_epoch(
        train_loader=train_loader,
        epoch=1,
        accumulation_steps=1,
        modality_weights={"atac": 1.0},
        resolution_weights={"atac": {1: 1.0, 128: 1.0}},
    )
    assert isinstance(train_loss, float)
    assert "atac" in per_modality

    val_loss, val_metrics = trainer.validate(
        val_loader=val_loader,
        modality_weights={"atac": 1.0},
        resolution_weights={"atac": {1: 1.0, 128: 1.0}},
        compute_pearson=True,
    )
    assert isinstance(val_loss, float)
    assert isinstance(val_metrics, dict)


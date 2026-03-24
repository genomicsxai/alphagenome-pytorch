"""Trainer API for AlphaGenome finetuning."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from alphagenome_pytorch.losses import multinomial_loss
from alphagenome_pytorch.extensions.finetuning.checkpointing import (
    load_checkpoint,
    save_checkpoint,
)
from alphagenome_pytorch.extensions.finetuning.distributed import (
    is_main_process,
    reduce_tensor,
)
from alphagenome_pytorch.extensions.finetuning.training import (
    NUM_SEGMENTS,
    validate,
)
from alphagenome_pytorch.extensions.finetuning.transfer import TransferConfig


@dataclass
class TrainerConfig:
    """Training hyperparameters for finetuning."""

    positional_weight: float = 5.0
    count_weight: float = 1.0
    max_grad_norm: float = 1.0
    num_segments: int = NUM_SEGMENTS
    min_segment_size: int | None = None
    use_amp: bool = True
    log_every: int = 50


class Trainer:
    """Training API for AlphaGenome finetuning.

    The trainer encapsulates model/heads/optimizer/scheduler and exposes:
    - forward_backward(): compute loss and accumulate gradients
    - optim_step(): clip gradients, step optimizer/scheduler, zero gradients
    - train_epoch()/validate(): convenience epoch methods
    """

    def __init__(
        self,
        model: nn.Module,
        heads: dict[str, nn.Module],
        optimizer: Optimizer,
        scheduler: LRScheduler,
        trainer_config: TrainerConfig,
        transfer_config: TransferConfig | None,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.model = model
        self.heads = heads
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = trainer_config
        self.transfer_config = transfer_config
        self.device = device
        self.rank = rank
        self.world_size = world_size

        self.model_module = self._unwrap_model(model)
        self.mode = self._resolve_mode(transfer_config)
        self.encoder_only = self.mode == "encoder-only"
        self.frozen_backbone = self._resolve_frozen_backbone(transfer_config, self.mode)

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        """Unwrap DDP and torch.compile wrappers to get the underlying model.

        This is important for checkpoint save/load to avoid serializing
        compiled wrappers or DDP key prefixes.
        """
        # Unwrap torch.compile (OptimizedModule) - may be nested
        while hasattr(model, "_orig_mod"):
            model = model._orig_mod
        # Unwrap DDP
        if hasattr(model, "module"):
            model = model.module
        return model

    @staticmethod
    def _resolve_mode(transfer_config: TransferConfig | None) -> str:
        if transfer_config is None:
            return "lora"
        mode = transfer_config.mode
        if isinstance(mode, list):
            # Most transfer use-cases are single mode; pick the explicit mode if present.
            if "encoder-only" in mode:
                return "encoder-only"
            if "linear-probe" in mode:
                return "linear-probe"
            if "linear" in mode:
                return "linear-probe"
            if "full" in mode:
                return "full"
            if "lora" in mode:
                return "lora"
            return mode[0]
        if mode == "linear":
            return "linear-probe"
        return mode

    @staticmethod
    def _resolve_frozen_backbone(transfer_config: TransferConfig | None, mode: str) -> bool:
        if mode in ("linear-probe", "encoder-only"):
            return True
        if mode == "lora":
            if transfer_config is None:
                return False
            return getattr(transfer_config, "lora_rank", 1) == 0
        return False

    @property
    def trainable_params(self) -> list[nn.Parameter]:
        params = [p for p in self.model.parameters() if p.requires_grad]
        for head in self.heads.values():
            params.extend([p for p in head.parameters() if p.requires_grad])
        return params

    def _amp_context(self):
        if self.config.use_amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @staticmethod
    def _compute_multinomial_resolution(
        seq_len: int,
        num_segments: int,
        min_segment_size: int | None,
    ) -> int:
        resolution = max(1, seq_len // num_segments)
        if min_segment_size is not None:
            resolution = max(resolution, min_segment_size)
        return resolution

    def _single_modality(self) -> str:
        if len(self.heads) != 1:
            raise ValueError("Single-modality payload provided but trainer has multiple heads.")
        return next(iter(self.heads.keys()))

    def _normalize_batch(
        self,
        batch: tuple[Tensor, dict[int, Tensor] | dict[str, dict[int, Tensor]]],
    ) -> tuple[Tensor, dict[str, dict[int, Tensor]]]:
        sequences, targets = batch
        if not isinstance(targets, dict):
            raise TypeError("Batch targets must be a dict.")
        if not targets:
            return sequences, {}

        first_key = next(iter(targets.keys()))
        if isinstance(first_key, int):
            modality = self._single_modality()
            return sequences, {modality: targets}  # type: ignore[arg-type]

        return sequences, targets  # type: ignore[return-value]

    def _default_resolution_weights(self) -> dict[str, dict[int, float]]:
        out: dict[str, dict[int, float]] = {}
        for modality, head in self.heads.items():
            out[modality] = {int(res): 1.0 for res in getattr(head, "resolutions", [128])}
        return out

    def _normalize_resolution_weights(
        self,
        resolution_weights: dict[str, dict[int, float]] | dict[int, float] | None,
    ) -> dict[str, dict[int, float]]:
        if resolution_weights is None:
            return self._default_resolution_weights()
        if not resolution_weights:
            return self._default_resolution_weights()

        first_key = next(iter(resolution_weights.keys()))
        if isinstance(first_key, int):
            modality = self._single_modality()
            return {modality: resolution_weights}  # type: ignore[arg-type]

        result = self._default_resolution_weights()
        for modality, weights in resolution_weights.items():  # type: ignore[assignment]
            result[modality] = {int(res): float(w) for res, w in weights.items()}
        return result

    def _normalize_modality_weights(
        self,
        modality_weights: dict[str, float] | None,
    ) -> dict[str, float]:
        if modality_weights is None:
            return {m: 1.0 for m in self.heads}
        out = {m: 1.0 for m in self.heads}
        for modality, w in modality_weights.items():
            out[modality] = float(w)
        return out

    def _forward_backbone(
        self,
        sequences: Tensor,
        all_resolutions: tuple[int, ...],
    ) -> dict[int, Tensor]:
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=self.device)

        if self.encoder_only:
            with torch.no_grad():
                with self._amp_context():
                    outputs = self.model(sequences, organism_idx, encoder_only=True)
            return {128: outputs["encoder_output"].detach()}

        if self.frozen_backbone:
            with torch.no_grad():
                with self._amp_context():
                    outputs = self.model(
                        sequences,
                        organism_idx,
                        return_embeddings=True,
                        resolutions=all_resolutions,
                        channels_last=False,
                    )
            embeddings: dict[int, Tensor] = {}
            for res in all_resolutions:
                emb_key = f"embeddings_{res}bp"
                if emb_key in outputs:
                    embeddings[res] = outputs[emb_key].detach()
            return embeddings

        with self._amp_context():
            outputs = self.model(
                sequences,
                organism_idx,
                return_embeddings=True,
                resolutions=all_resolutions,
                channels_last=False,
            )
        embeddings = {}
        for res in all_resolutions:
            emb_key = f"embeddings_{res}bp"
            if emb_key in outputs:
                embeddings[res] = outputs[emb_key]
        return embeddings

    def forward_backward(
        self,
        batch: tuple[Tensor, dict[int, Tensor] | dict[str, dict[int, Tensor]]],
        modality_weights: dict[str, float] | None = None,
        resolution_weights: dict[str, dict[int, float]] | dict[int, float] | None = None,
        accumulation_steps: int = 1,
    ) -> dict[str, float]:
        """Compute loss and accumulate gradients (without stepping optimizer)."""
        self.model.train()
        for head in self.heads.values():
            head.train()

        sequences, modality_targets = self._normalize_batch(batch)
        sequences = sequences.to(self.device)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=self.device)

        modality_weights = self._normalize_modality_weights(modality_weights)
        resolution_weights = self._normalize_resolution_weights(resolution_weights)

        all_resolutions = sorted({res for weights in resolution_weights.values() for res in weights.keys()})
        embeddings_dict = self._forward_backbone(sequences, tuple(all_resolutions))

        total_loss = torch.tensor(0.0, device=self.device)
        metrics: dict[str, float] = {}

        single_modality = len(self.heads) == 1
        only_modality = next(iter(self.heads.keys())) if single_modality else None

        for modality, head in self.heads.items():
            if modality not in modality_targets:
                continue

            targets_dict = modality_targets[modality]
            res_weights = resolution_weights.get(modality, {})
            modality_weight = modality_weights.get(modality, 1.0)

            with self._amp_context():
                predictions = head(
                    embeddings_dict,
                    organism_idx,
                    return_scaled=True,
                    channels_last=True,
                )

            modality_loss = torch.tensor(0.0, device=self.device)
            head_module = head.module if hasattr(head, "module") else head

            for res, res_weight in res_weights.items():
                if res not in predictions or res not in targets_dict:
                    continue

                pred = predictions[res]
                targets = targets_dict[res].to(self.device)
                targets_scaled = head_module.scale(
                    targets,
                    organism_idx,
                    resolution=res,
                    channels_last=True,
                )

                mask = torch.ones(
                    pred.shape[0], 1, pred.shape[-1], dtype=torch.bool, device=self.device
                )
                multinomial_res = self._compute_multinomial_resolution(
                    pred.shape[-2],
                    self.config.num_segments,
                    self.config.min_segment_size,
                )
                loss_dict = multinomial_loss(
                    y_pred=pred,
                    y_true=targets_scaled,
                    mask=mask,
                    multinomial_resolution=multinomial_res,
                    positional_weight=self.config.positional_weight,
                    count_weight=self.config.count_weight,
                    channels_last=True,
                )

                res_loss = loss_dict["loss"] * float(res_weight)
                modality_loss = modality_loss + res_loss

                metrics[f"{modality}_loss_{res}bp"] = res_loss.item()
                metrics[f"{modality}_loss_{res}bp_count"] = loss_dict["loss_total"].item()
                metrics[f"{modality}_loss_{res}bp_positional"] = loss_dict["loss_positional"].item()

                if single_modality and only_modality == modality:
                    metrics[f"loss_{res}bp"] = res_loss.item()
                    metrics[f"loss_{res}bp_count"] = loss_dict["loss_total"].item()
                    metrics[f"loss_{res}bp_positional"] = loss_dict["loss_positional"].item()

            weighted_modality_loss = modality_loss * float(modality_weight)
            total_loss = total_loss + weighted_modality_loss
            metrics[f"{modality}_loss"] = modality_loss.item()

        if not total_loss.requires_grad:
            raise RuntimeError(
                "No valid losses were computed for this batch. "
                "Check modality and resolution weights against batch targets."
            )

        (total_loss / accumulation_steps).backward()
        metrics["loss"] = total_loss.item()
        return metrics

    def optim_step(self) -> dict[str, float]:
        """Clip gradients and step optimizer/scheduler."""
        params = self.trainable_params
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return {
            "grad_norm": float(grad_norm),
            "learning_rate": float(self.scheduler.get_last_lr()[0]),
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        accumulation_steps: int = 1,
        modality_weights: dict[str, float] | None = None,
        resolution_weights: dict[str, dict[int, float]] | dict[int, float] | None = None,
        train_sampler: DistributedSampler | None = None,
        log_fn: Callable[[dict[str, float]], None] | None = None,
        profile_batches: int = 0,
    ) -> tuple[float, dict[str, float]]:
        """Train one epoch using forward_backward() + optim_step()."""
        if profile_batches > 0 and is_main_process(self.rank):
            print(
                "Warning: profile_batches is currently ignored in Trainer.train_epoch. "
                "Use the legacy training functions if detailed profiling is required."
            )

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main_process(self.rank):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        else:
            pbar = train_loader

        total_loss = 0.0
        n_batches = 0
        running_loss = 0.0
        running_batches = 0
        per_modality_total = {m: 0.0 for m in self.heads}

        for batch_idx, batch in enumerate(pbar):
            losses = self.forward_backward(
                batch=batch,
                modality_weights=modality_weights,
                resolution_weights=resolution_weights,
                accumulation_steps=accumulation_steps,
            )

            is_accumulation_step = (batch_idx + 1) % accumulation_steps == 0
            is_last_batch = batch_idx == len(train_loader) - 1
            if is_accumulation_step or is_last_batch:
                step_metrics = self.optim_step()
            else:
                step_metrics = {"learning_rate": float(self.scheduler.get_last_lr()[0])}

            raw_loss = float(losses["loss"])
            total_loss += raw_loss
            n_batches += 1
            running_loss += raw_loss
            running_batches += 1

            for modality in per_modality_total:
                per_modality_total[modality] += float(losses.get(f"{modality}_loss", 0.0))

            if is_main_process(self.rank) and batch_idx % self.config.log_every == 0:
                avg_running_loss = running_loss / max(1, running_batches)
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix({
                        "loss": f"{raw_loss:.4f}",
                        "run_loss": f"{avg_running_loss:.4f}",
                        "lr": f"{step_metrics['learning_rate']:.2e}",
                    })

                if log_fn is not None:
                    step_log = {
                        "batch": float(batch_idx),
                        "epoch": float(epoch),
                        "loss": raw_loss,
                        "running_loss": avg_running_loss,
                        "learning_rate": float(step_metrics["learning_rate"]),
                    }
                    step_log.update(losses)
                    log_fn(step_log)

                running_loss = 0.0
                running_batches = 0

        avg_loss = total_loss / max(1, n_batches)
        per_modality_loss = {
            modality: total / max(1, n_batches)
            for modality, total in per_modality_total.items()
        }

        if self.world_size > 1:
            avg_tensor = torch.tensor(avg_loss, device=self.device)
            avg_tensor = reduce_tensor(avg_tensor, self.world_size)
            avg_loss = avg_tensor.item()

            for modality, val in list(per_modality_loss.items()):
                m_tensor = torch.tensor(val, device=self.device)
                m_tensor = reduce_tensor(m_tensor, self.world_size)
                per_modality_loss[modality] = m_tensor.item()

        return avg_loss, per_modality_loss

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        modality_weights: dict[str, float] | None = None,
        resolution_weights: dict[str, dict[int, float]] | dict[int, float] | None = None,
        compute_pearson: bool = True,
    ) -> tuple[float, dict[str, Any]]:
        """Validation pass using legacy-validated metric implementations."""
        resolution_weights_norm = self._normalize_resolution_weights(resolution_weights)
        modality_weights_norm = self._normalize_modality_weights(modality_weights)

        # Gate use_amp on CUDA device - validate uses torch.autocast(device_type='cuda')
        use_amp = self.config.use_amp and self.device.type == "cuda"

        return validate(
            model=self.model,
            heads=self.heads,
            val_loader=val_loader,
            device=self.device,
            modality_weights=modality_weights_norm,
            resolution_weights=resolution_weights_norm,
            positional_weight=self.config.positional_weight,
            count_weight=self.config.count_weight,
            use_amp=use_amp,
            num_segments=self.config.num_segments,
            min_segment_size=self.config.min_segment_size,
            compute_pearson=compute_pearson,
            rank=self.rank,
            world_size=self.world_size,
            encoder_only=self.encoder_only,
        )

    @torch.no_grad()
    def predict(
        self,
        sequences: Tensor,
        modalities: list[str] | None = None,
        resolutions: tuple[int, ...] | None = None,
    ) -> dict[str, dict[int, Tensor]]:
        """Run inference for selected modalities and resolutions."""
        self.model.eval()
        for head in self.heads.values():
            head.eval()

        selected_modalities = modalities if modalities is not None else list(self.heads.keys())
        sequences = sequences.to(self.device)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=self.device)

        if resolutions is None:
            resolution_set: set[int] = set()
            for modality in selected_modalities:
                resolution_set.update(getattr(self.heads[modality], "resolutions", [128]))
        else:
            resolution_set = set(resolutions)

        if self.encoder_only:
            with self._amp_context():
                outputs = self.model(sequences, organism_idx, encoder_only=True)
            embeddings_dict: dict[int, Tensor] = {128: outputs["encoder_output"]}
        else:
            with self._amp_context():
                outputs = self.model(
                    sequences,
                    organism_idx,
                    return_embeddings=True,
                    resolutions=tuple(sorted(resolution_set)),
                    channels_last=False,
                )

            embeddings_dict = {}
            for res in sorted(resolution_set):
                emb_key = f"embeddings_{res}bp"
                if emb_key in outputs:
                    embeddings_dict[res] = outputs[emb_key]

        predictions: dict[str, dict[int, Tensor]] = {}
        for modality in selected_modalities:
            head = self.heads[modality]
            with self._amp_context():
                modality_preds = head(
                    embeddings_dict,
                    organism_idx,
                    return_scaled=False,
                    channels_last=True,
                )
            if resolutions is not None:
                modality_preds = {res: pred for res, pred in modality_preds.items() if res in resolution_set}
            predictions[modality] = modality_preds

        return predictions

    def _default_track_names(self) -> list[str] | dict[str, list[str]]:
        if len(self.heads) == 1:
            head = next(iter(self.heads.values()))
            n_tracks = int(getattr(head, "num_tracks", 0))
            return [f"track_{i}" for i in range(n_tracks)]

        out: dict[str, list[str]] = {}
        for modality, head in self.heads.items():
            n_tracks = int(getattr(head, "num_tracks", 0))
            out[modality] = [f"{modality}_track_{i}" for i in range(n_tracks)]
        return out

    def _default_modalities(self) -> str | list[str]:
        keys = list(self.heads.keys())
        if len(keys) == 1:
            return keys[0]
        return keys

    def _default_resolutions(self) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
        if len(self.heads) == 1:
            head = next(iter(self.heads.values()))
            return tuple(int(r) for r in getattr(head, "resolutions", [128]))
        return {
            modality: tuple(int(r) for r in getattr(head, "resolutions", [128]))
            for modality, head in self.heads.items()
        }

    def save_state(
        self,
        path: Path | str,
        epoch: int = 0,
        val_loss: float = 0.0,
        track_names: list[str] | dict[str, list[str]] | None = None,
        modality: str | list[str] | None = None,
        resolutions: tuple[int, ...] | dict[str, tuple[int, ...]] | None = None,
        best_val_loss: float | None = None,
        wandb_run_id: str | None = None,
        **extra_metadata: Any,
    ) -> None:
        """Save full trainer state via the shared checkpointing utility."""
        if track_names is None:
            track_names = self._default_track_names()
        if modality is None:
            modality = self._default_modalities()
        if resolutions is None:
            resolutions = self._default_resolutions()

        save_checkpoint(
            path=path,
            epoch=epoch,
            model=self.model_module,
            optimizer=self.optimizer,
            val_loss=val_loss,
            track_names=track_names,
            modality=modality,
            resolutions=resolutions,
            scheduler=self.scheduler,
            best_val_loss=best_val_loss,
            wandb_run_id=wandb_run_id,
            trainer_config=asdict(self.config),
            transfer_mode=self.mode,
            schema_version=2,
            **extra_metadata,
        )

    def load_state(self, path: Path | str, device: str = "cpu") -> dict[str, Any]:
        """Load trainer state and return checkpoint metadata."""
        return load_checkpoint(
            path=path,
            model=self.model_module,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=device,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: Path | str,
        model: nn.Module,
        heads: dict[str, nn.Module],
        optimizer: Optimizer,
        scheduler: LRScheduler,
        trainer_config: TrainerConfig,
        transfer_config: TransferConfig | None,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        load_device: str = "cpu",
    ) -> tuple["Trainer", dict[str, Any]]:
        """Construct trainer and load checkpoint state."""
        trainer = cls(
            model=model,
            heads=heads,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_config=trainer_config,
            transfer_config=transfer_config,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        metadata = trainer.load_state(path, device=load_device)
        return trainer, metadata


__all__ = ["Trainer", "TrainerConfig"]

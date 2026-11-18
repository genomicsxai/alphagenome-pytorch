"""Training logging utilities for AlphaGenome fine-tuning.

Provides TrainingLogger for CSV and optional W&B logging with rank awareness.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from alphagenome_pytorch.extensions.finetuning.distributed import is_main_process


class TrainingLogger:
    """Logger for training metrics with optional W&B integration.

    Handles both step-level (batch) and epoch-level logging to CSV files
    and optionally to Weights & Biases. Only logs on rank 0 in distributed mode.

    Attributes:
        output_dir: Directory for log files.
        rank: Process rank (0 for main process).
        use_wandb: Whether W&B logging is enabled.
        step: Current step counter.

    Example:
        >>> logger = TrainingLogger(
        ...     output_dir=Path("output"),
        ...     rank=0,
        ...     use_wandb=True,
        ...     wandb_project="my-project",
        ...     config={"lr": 1e-4, "epochs": 10},
        ... )
        >>> logger.log_step({"loss": 0.5, "lr": 1e-4})
        >>> logger.log_epoch(1, train_loss=0.5, val_loss=0.4, lr=1e-4)
        >>> logger.finish()
    """

    def __init__(
        self,
        output_dir: Path,
        rank: int = 0,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        run_name: str | None = None,
        config: dict | None = None,
        resume_id: str | None = None,
    ) -> None:
        """Initialize the training logger.

        Args:
            output_dir: Directory for log files (created if it doesn't exist).
            rank: Process rank for distributed training. Only rank 0 logs.
            use_wandb: Whether to enable Weights & Biases logging.
            wandb_project: W&B project name.
            wandb_entity: W&B entity (team/user).
            run_name: Name for this run.
            config: Configuration dict to save and log to W&B.
            resume_id: W&B run ID for resuming a previous run.
        """
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.use_wandb = use_wandb and is_main_process(rank)
        self.step = 0
        self.resume_id = resume_id

        # Only main process handles logging
        if not is_main_process(rank):
            self.csv_file = None
            self.csv_writer = None
            self._csv_fieldnames = None
            self.wandb = None
            return

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CSV logging
        self.csv_path = self.output_dir / "training_log.csv"
        self.csv_file = None
        self.csv_writer = None
        self._csv_fieldnames: list[str] | None = None

        # Save config
        if config:
            config_path = self.output_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, default=str)
            print(f"Saved config to {config_path}")

        # Initialize W&B if requested
        if self.use_wandb:
            try:
                import wandb

                self.wandb = wandb
                wandb.init(
                    project=wandb_project or "alphagenome-finetune",
                    entity=wandb_entity,
                    name=run_name,
                    config=config,
                    dir=str(self.output_dir),
                    id=resume_id,
                    resume="allow" if resume_id else None,
                )
                if resume_id and wandb.run is not None:
                    # wandb restored its internal step counter from the prior
                    # run; sync ours so subsequent log_step/log_epoch calls
                    # produce monotonically increasing steps.
                    self.step = int(wandb.run.step or 0)
                print(f"W&B initialized: {wandb.run.url}" + (" (resumed)" if resume_id else ""))
            except ImportError:
                print("Warning: wandb not installed, disabling W&B logging")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

    def _ensure_csv(self, fieldnames: list[str]) -> None:
        """Initialize CSV file with headers if not already done."""
        if not is_main_process(self.rank):
            return
        if self.csv_writer is None:
            self._csv_fieldnames = fieldnames
            # Append mode to support resume
            write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            self.csv_file = open(self.csv_path, "a", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            if write_header:
                self.csv_writer.writeheader()
            self.csv_file.flush()

    def log_step(self, metrics: dict[str, Any]) -> None:
        """Log metrics for a training step.

        Args:
            metrics: Dictionary of metric name -> value.
                     'step' and 'timestamp' are added automatically.
        """
        if not is_main_process(self.rank):
            return

        self.step += 1
        metrics["step"] = self.step
        metrics["timestamp"] = datetime.now().isoformat()

        # CSV logging
        fieldnames = ["step", "timestamp"] + [
            k for k in sorted(metrics.keys()) if k not in ["step", "timestamp"]
        ]
        self._ensure_csv(fieldnames)

        # Only write fields that exist in the header
        row = {k: v for k, v in metrics.items() if k in self._csv_fieldnames}
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        # W&B logging
        if self.use_wandb:
            self.wandb.log(metrics, step=self.step)

    def _append_epoch_scalars(self, metrics: dict[str, Any]) -> None:
        path = self.output_dir / "epoch_log.csv"
        fieldnames = []
        if path.exists():
            with path.open(newline="") as existing:
                fieldnames = next(csv.reader(existing), [])
        new_columns = [key for key in metrics if key not in fieldnames]
        if fieldnames and new_columns:
            # Rewrite once when a new metric appears, preserving old rows by
            # column name. Replace only after the new file has been written.
            fieldnames += new_columns
            with NamedTemporaryFile("w", dir=path.parent, newline="", delete=False) as file:
                temporary = Path(file.name)
                try:
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()
                    with path.open(newline="") as existing:
                        writer.writerows(csv.DictReader(existing))
                    writer.writerow(metrics)
                    file.close()
                    temporary.chmod(path.stat().st_mode & 0o777)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            return

        with path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames or new_columns)
            if not fieldnames:
                writer.writeheader()
            writer.writerow(metrics)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        is_best: bool = False,
        extra: dict[str, Any] | None = None,
        histograms: dict[str, list[float]] | None = None,
        *,
        track_metrics: dict[str, list[tuple[int, str, float]]] | None = None,
    ) -> None:
        """Log epoch-level metrics.

        Args:
            epoch: Current epoch number.
            train_loss: Training loss for this epoch.
            val_loss: Validation loss for this epoch.
            lr: Current learning rate.
            is_best: Whether this is the best model so far.
            extra: Additional scalar metrics to log.
            histograms: Dict of metric_name -> list of values for histogram logging
                       (only logged to W&B, not CSV).
            track_metrics: Per-metric rows of (track index, track name, value).
        """
        if not is_main_process(self.rank):
            return

        metrics: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": lr,
            "is_best": is_best,
            "timestamp": datetime.now().isoformat(),
        }
        if extra:
            metrics.update(extra)

        self._append_epoch_scalars(metrics)
        if track_metrics:
            path = self.output_dir / "epoch_track_metrics.csv"
            write_header = not path.exists() or path.stat().st_size == 0
            with path.open("a", newline="") as file:
                writer = csv.writer(file)
                if write_header:
                    writer.writerow(("epoch", "metric", "track_index", "track_name", "value"))
                for key, tracks in track_metrics.items():
                    writer.writerows((epoch, key, index, name, value) for index, name, value in tracks)

        # W&B logging
        if self.use_wandb:
            wandb_metrics: dict[str, Any] = {
                "epoch": epoch,
                "epoch/train_loss": train_loss,
                "epoch/val_loss": val_loss,
                "epoch/learning_rate": lr,
            }
            if extra:
                for k, v in extra.items():
                    wandb_metrics[f"epoch/{k}"] = v
            for key, tracks in (track_metrics or {}).items():
                if f"{key}_per_track_values" in (histograms or {}):
                    continue
                names = [name for _, name, _ in tracks]
                for index, name, value in tracks:
                    label = name if names.count(name) == 1 else f"{name} [{index}]"
                    wandb_metrics[f"epoch/{key}/{label}"] = value
            # Log histograms for distributions
            if histograms:
                for k, values in histograms.items():
                    finite_values = [value for value in values if isfinite(value)]
                    if finite_values:
                        wandb_metrics[f"epoch/{k}"] = self.wandb.Histogram(finite_values)
            self.wandb.log(wandb_metrics, step=self.step)

    @property
    def wandb_run_id(self) -> str | None:
        """Get the current W&B run ID for checkpoint saving."""
        if self.use_wandb and self.wandb and self.wandb.run:
            return self.wandb.run.id
        return None

    def finish(self) -> None:
        """Close logger and finalize W&B run."""
        if self.csv_file:
            self.csv_file.close()
        if self.use_wandb and self.wandb:
            self.wandb.finish()


__all__ = ["TrainingLogger"]

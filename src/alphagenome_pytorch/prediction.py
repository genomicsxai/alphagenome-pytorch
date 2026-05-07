"""Shared raw prediction runtime for AlphaGenome models."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Mapping

import torch

from alphagenome_pytorch.genome import (
    GenomeSequenceSource,
    Interval,
    apply_variant_to_sequence,
)
from alphagenome_pytorch.named_outputs import TrackMetadata, TrackMetadataCatalog
from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor


class AlphaGenomePredictionRuntime:
    """Own model/device/FASTA plumbing for raw prediction calls.

    This is intentionally independent from variant scoring. Scoring can compose
    this runtime, but raw serving and attribution should not need to construct a
    ``VariantScoringModel``.
    """

    organism_map = {
        "human": 0,
        "homo_sapiens": 0,
        "mouse": 1,
        "mus_musculus": 1,
        "HOMO_SAPIENS": 0,
        "MUS_MUSCULUS": 1,
        # NCBI taxonomy IDs — these are the values of the
        # ``dna_model_pb2.ORGANISM_*`` proto enum constants, so the runtime can
        # accept them transparently without importing alphagenome.protos.
        9606: 0,
        10090: 1,
    }

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        fasta_path: str | Path | None = None,
        sequence_source: GenomeSequenceSource | None = None,
        metadata_catalog: TrackMetadataCatalog | None = None,
        track_metadata: Mapping[int, Mapping[Any, list[Any]]] | None = None,
        track_names: Mapping[str, list[str]] | list[str] | None = None,
        device: str | torch.device | None = None,
        default_organism: str | int | None = "human",
    ):
        if sequence_source is None and fasta_path is not None:
            sequence_source = GenomeSequenceSource(fasta_path)
        self.sequence_source = sequence_source
        self.metadata_catalog = metadata_catalog
        self._legacy_track_metadata = track_metadata or {}
        self._track_names = track_names

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.num_organisms = getattr(model, "num_organisms", 2)
        self.default_organism_index = (
            self.resolve_organism_index(default_organism)
            if default_organism is not None
            else None
        )

    def resolve_organism_index(self, organism: Any = None) -> int:
        """Map an organism specifier to the runtime's internal organism index.

        Accepts:

        * ``None`` — returns ``self.default_organism_index`` (or 0 if unset).
        * Strings: ``"human"``, ``"mouse"``, ``"homo_sapiens"``, ``"HOMO_SAPIENS"``,
          ``"ORGANISM_HOMO_SAPIENS"`` (proto enum name), etc.
        * Integers: NCBI taxonomy IDs 9606 / 10090 (the values of the
          ``dna_model_pb2.ORGANISM_*`` proto enum constants), or raw 0/1
          internal indices via the ``__index__`` fallback.
        * Enum-like objects exposing ``.value`` or ``.name``.

        Raw integers fall through to a bounds check against
        ``self.num_organisms``.
        """
        if organism is None:
            return self.default_organism_index if self.default_organism_index is not None else 0
        if hasattr(organism, "value"):
            candidate = getattr(organism, "value")
            if candidate in self.organism_map:
                return self.organism_map[candidate]
        if hasattr(organism, "name"):
            candidate = getattr(organism, "name")
            if candidate in self.organism_map:
                return self.organism_map[candidate]
            if isinstance(candidate, str):
                stripped = candidate.removeprefix("ORGANISM_")
                if stripped in self.organism_map:
                    return self.organism_map[stripped]
        if organism in self.organism_map:
            return self.organism_map[organism]
        if hasattr(organism, "__index__"):
            idx = int(organism)
        elif isinstance(organism, str):
            normalized = organism.upper().removeprefix("ORGANISM_")
            if normalized in self.organism_map:
                idx = self.organism_map[normalized]
            else:
                lower = organism.lower()
                if lower in self.organism_map:
                    idx = self.organism_map[lower]
                else:
                    idx = int(organism)
        else:
            raise ValueError(f"Invalid organism type: {type(organism)}")
        if idx < 0 or idx >= self.num_organisms:
            raise ValueError(
                f"Organism index {idx} out of range for model with {self.num_organisms} organisms"
            )
        return idx

    def get_sequence(self, interval: Any, variant: Any | None = None) -> str:
        if self.sequence_source is None:
            raise ValueError("FASTA path not provided. Cannot extract interval sequence.")
        return self.sequence_source.fetch_sequence(interval, variant=variant)

    @torch.no_grad()
    def predict(
        self,
        sequence: str | torch.Tensor,
        organism: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        organism_index = self.resolve_organism_index(organism)
        if isinstance(sequence, str):
            dtype = getattr(self.model, "dtype_policy", None)
            dtype = getattr(dtype, "compute_dtype", torch.float32)
            onehot = sequence_to_onehot_tensor(
                sequence,
                dtype=dtype,
                device=self.device,
                ambiguous="zero",
            )
        else:
            dtype = getattr(self.model, "dtype_policy", None)
            dtype = getattr(dtype, "compute_dtype", sequence.dtype)
            onehot = sequence.to(dtype=dtype, device=self.device)
        if onehot.dim() == 2:
            onehot = onehot.unsqueeze(0)
        org_idx = torch.full(
            (onehot.shape[0],),
            organism_index,
            dtype=torch.long,
            device=self.device,
        )
        return self.model(onehot, org_idx, **kwargs)

    @torch.no_grad()
    def predict_variant(
        self,
        interval: Any,
        variant: Any,
        organism: Any = None,
        *,
        to_cpu: bool = False,
        unified_splicing: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if unified_splicing:
            raise NotImplementedError(
                "Unified splicing prediction remains part of variant scoring."
            )
        interval_length = int(getattr(interval, "width", getattr(interval, "end") - getattr(interval, "start")))
        deletion_extension = max(
            0,
            len(getattr(variant, "reference_bases")) - len(getattr(variant, "alternate_bases")),
        )
        if deletion_extension > 0:
            extraction_interval = Interval(
                getattr(interval, "chromosome"),
                int(getattr(interval, "start")),
                int(getattr(interval, "end")) + deletion_extension,
                getattr(interval, "strand", "."),
                getattr(interval, "name", ""),
            )
        else:
            extraction_interval = interval

        base_seq = self.get_sequence(extraction_interval)
        ref_seq = base_seq[:interval_length]
        alt_seq = apply_variant_to_sequence(base_seq, variant, extraction_interval)[:interval_length]
        ref_outputs = self.predict(ref_seq, organism)
        alt_outputs = self.predict(alt_seq, organism)
        if to_cpu:
            ref_outputs = self._outputs_to_cpu(ref_outputs)
            alt_outputs = self._outputs_to_cpu(alt_outputs)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return ref_outputs, alt_outputs

    def get_track_metadata(self, organism: Any = None, output_name: str | None = None) -> list[Any] | dict[Any, list[Any]]:
        organism_index = self.resolve_organism_index(organism)
        if output_name is not None and self.metadata_catalog is not None:
            return list(self.metadata_catalog.get_tracks(output_name, organism=organism_index))
        legacy = self._legacy_track_metadata.get(organism_index, {})
        if output_name is None:
            return legacy
        for key, value in legacy.items():
            key_name = getattr(key, "value", str(key))
            if key_name == output_name:
                return value
        names = self._track_names_for_output(output_name)
        if names:
            return [
                TrackMetadata(
                    track_index=i,
                    output_name=output_name,
                    organism=organism_index,
                    track_name=name,
                )
                for i, name in enumerate(names)
            ]
        return []

    def _track_names_for_output(self, output_name: str) -> list[str] | None:
        if self._track_names is None:
            return None
        if isinstance(self._track_names, Mapping):
            return self._track_names.get(output_name)
        return self._track_names

    def _outputs_to_cpu(self, outputs: Any) -> Any:
        if torch.is_tensor(outputs):
            return outputs.cpu()
        if isinstance(outputs, dict):
            return {k: self._outputs_to_cpu(v) for k, v in outputs.items()}
        if isinstance(outputs, list):
            return [self._outputs_to_cpu(v) for v in outputs]
        if isinstance(outputs, tuple):
            return tuple(self._outputs_to_cpu(v) for v in outputs)
        return outputs


__all__ = ["AlphaGenomePredictionRuntime"]

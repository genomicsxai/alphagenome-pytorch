"""Candidate provenance must drive both head positions and target selection."""
import ast
import symtable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from alphagenome_pytorch.extensions.finetuning import runner, training, star_junctions


@pytest.mark.parametrize("capacity", [256, 512])
@pytest.mark.parametrize("source", ["annotated", "predicted"])
def test_source_controls_prediction_top_k(source, capacity):
    resolver = getattr(runner, "junction_prediction_top_k", None)
    assert callable(resolver), "Runner must explicitly resolve annotated vs predicted routing"
    assert resolver(source, capacity) == (capacity if source == "predicted" else None)


@pytest.mark.parametrize("source,capacity", [("unknown", 256), ("predicted", 0), ("predicted", -1)])
def test_invalid_prediction_settings_fail(source, capacity):
    resolver = getattr(runner, "junction_prediction_top_k", None)
    assert callable(resolver)
    with pytest.raises(ValueError):
        resolver(source, capacity)


def test_all_runner_train_and_eval_paths_use_resolved_top_k():
    tree = ast.parse(Path(runner.__file__).read_text())
    values = [kw.value for node in ast.walk(tree) if isinstance(node, ast.Call)
              for kw in node.keywords if kw.arg == "junction_top_k"]
    assert len(values) == 4  # eval-only, SP train, standard train, validation
    assert all(isinstance(value, ast.Name) and value.id == "junction_forward_top_k"
               for value in values)


@pytest.mark.parametrize("source", ["annotated", "predicted"])
def test_resolved_top_k_is_initialized_in_main_before_use(source):
    # Checking only call-argument names misses an assignment accidentally
    # inserted into another function. Check scope and execute the real setup
    # statement without initializing a model or loading any biological data.
    code = Path(runner.__file__).read_text()
    tree = ast.parse(code)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "main")
    scope = next(child for child in symtable.symtable(code, runner.__file__, "exec").get_children()
                 if child.get_name() == "main")
    assert scope.lookup("junction_forward_top_k").is_local()
    assignments = [node for node in main.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "junction_forward_top_k"
                           for target in node.targets)]
    assert len(assignments) == 1
    uses = [node.lineno for node in ast.walk(main) if isinstance(node, ast.Name)
            and node.id == "junction_forward_top_k" and isinstance(node.ctx, ast.Load)]
    assert assignments[0].lineno < min(uses)
    namespace = {"args": SimpleNamespace(junction_position_source=source, junction_top_k=256),
                 "junction_prediction_top_k": runner.junction_prediction_top_k}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), runner.__file__, "exec"), namespace)
    assert namespace["junction_forward_top_k"] == (None if source == "annotated" else 256)


def test_track_metadata_loader_does_not_depend_on_cli_args(monkeypatch):
    # Exercise the real helper's nonempty-path branch, where an accidentally
    # inserted reference to main's args would fail before reading the catalog.
    reached = Mock(side_effect=ValueError("catalog reached"))
    monkeypatch.setattr(runner.TrackMetadataCatalog, "from_file", reached)
    with pytest.raises(ValueError, match="catalog reached"):
        runner.load_track_metadata_for_finetune("fixture.parquet", {}, rank=0, organism="human")
    reached.assert_called_once_with("fixture.parquet", default_organism=0)


@pytest.mark.parametrize("source", ["annotated", "predicted"])
def test_main_resolves_routing_before_loading_base_weights(monkeypatch, tmp_path, source):
    args = SimpleNamespace(seed=None, run_name="routing-smoke", output_dir=str(tmp_path),
                           resume=None, track_metadata=None, organism="human",
                           junction_position_source=source, junction_top_k=256,
                           max_splice_sites=256,
                           pretrained_weights="unused.safetensors")
    monkeypatch.setattr(runner, "setup_distributed", lambda: (0, 1, 0, torch.device("cpu")))
    monkeypatch.setattr(runner, "barrier", lambda: None)
    monkeypatch.setattr(runner, "create_datasets", lambda *args: (None, None, {}, {}))
    resolver = Mock(wraps=runner.junction_prediction_top_k)
    monkeypatch.setattr(runner, "junction_prediction_top_k", resolver)
    # Stop at the first weight I/O boundary: invoke actual main startup without
    # allocating the large model or reading a genome/checkpoint.
    stop = Mock(side_effect=ValueError("base-weight boundary reached"))
    monkeypatch.setattr(runner, "compute_base_model_weights_hash_from_file", stop)
    with pytest.raises(ValueError, match="base-weight boundary reached"):
        runner.main(args)
    resolver.assert_called_once_with(source, 256)
    stop.assert_called_once_with("unused.safetensors")


class FakeHead:
    _num_tissues = 2

    def __call__(self, embeddings, organism, *, splice_site_positions, channels_last):
        self.seen_positions = splice_site_positions
        batch, _, capacity = splice_site_positions.shape
        return {"pred_counts": torch.ones(batch, capacity, capacity, 4)}

    def conv(self, embeddings, organism):
        return embeddings

    def _predict_from_sparse_logits(self, *args):
        positions = args[-2]
        self.seen_positions = positions
        batch, _, capacity = positions.shape
        return torch.ones(batch, capacity, capacity, 4), None


@pytest.mark.parametrize("predicted", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_head_routing_emits_explicit_provenance(monkeypatch, predicted, sequence_parallel):
    monkeypatch.setattr(training, "SpliceSitesJunctionHead", FakeHead)
    head = FakeHead()
    embeddings = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    annotated = torch.tensor([[[1, 3, -1]] * 4])
    predicted_positions = torch.tensor([[[7, 6]] * 4])
    classifier = Mock()
    classifier.return_value = {"logits": torch.zeros(1, 5, 8)}
    classifier.conv.return_value = torch.zeros(1, 5, 8)
    selector = Mock(return_value=predicted_positions)
    monkeypatch.setattr(training, "_top_k_positions_from_logits", selector)
    top_k = 2 if predicted else None
    if sequence_parallel:
        sp = SimpleNamespace(
            gather_full=lambda tensor, **kwargs: tensor,
            gather_positions=lambda tensor, *, global_indices, **kwargs: tensor[..., global_indices],
        )
        result = training._call_splice_junction_head_sp(
            head, embeddings, annotated, torch.zeros(1, dtype=torch.long),
            sp, 8, top_k, classifier, torch.device("cpu"),
        )
    else:
        result = training._call_splice_head(
            head, {1: embeddings}, torch.zeros(1, dtype=torch.long),
            annotated, False, cls_head=classifier, junction_top_k=top_k,
        )
    assert result.get("positions_are_predicted") is predicted
    expected = predicted_positions if predicted else annotated
    torch.testing.assert_close(result["positions"], expected)
    assert result["pos_counts"].shape == (1, expected.shape[-1], expected.shape[-1], 2)
    if predicted:
        selector.assert_called_once()
        assert selector.call_args.args[1] == 2
    else:
        selector.assert_not_called()
        classifier.assert_not_called()
        classifier.conv.assert_not_called()


def test_annotated_targets_do_not_rebuild_even_with_positions_key(monkeypatch):
    positions = torch.tensor([[[1, 3, -1]] * 4])
    matrix = torch.arange(36, dtype=torch.float32).reshape(1, 3, 3, 4)
    builder = Mock(side_effect=AssertionError("Annotated targets must not be rebuilt"))
    monkeypatch.setattr(star_junctions, "junctions_to_junction_matrix", builder)
    actual, actual_positions = training._get_junction_targets(
        {"positions": positions, "positions_are_predicted": False},
        {"junction_matrix": matrix, "junction_positions": positions}, torch.device("cpu"),
    )
    torch.testing.assert_close(actual, matrix)
    torch.testing.assert_close(actual_positions, positions)
    builder.assert_not_called()


def test_predicted_targets_rebuild_at_predicted_coordinates(monkeypatch):
    positions = torch.tensor([[[7, 6]] * 4])
    matrix = np.arange(16, dtype=np.float32).reshape(2, 2, 4)
    builder = Mock(return_value=(None, matrix))
    monkeypatch.setattr(star_junctions, "junctions_to_junction_matrix", builder)
    actual, actual_positions = training._get_junction_targets(
        {"positions": positions, "positions_are_predicted": True},
        {"all_junctions": [["sample fixture"]]}, torch.device("cpu"),
    )
    torch.testing.assert_close(actual, torch.from_numpy(matrix)[None])
    torch.testing.assert_close(actual_positions, positions)
    builder.assert_called_once()
    assert builder.call_args.kwargs["max_splice_sites"] == 2
    np.testing.assert_array_equal(builder.call_args.kwargs["positions"], positions[0].numpy())

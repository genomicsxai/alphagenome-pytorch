"""Tests for converting alphagenome_ft (JAX) finetuned checkpoints to PyTorch deltas."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from alphagenome_pytorch.jax_compat.finetuned import (
    build_finetuned_conversion,
    convert_finetuned_checkpoint,
    is_alphagenome_ft_checkpoint,
    parse_track_means,
)

RANK, ALPHA = 4, 8


def _track_head_config(name="rna_seq", output_type="RNA_SEQ", resolutions=(1,), squash=True):
    return {
        "source": "predefined", "type": "GENOME_TRACKS", "name": name,
        "output_type": output_type, "resolutions": list(resolutions), "apply_squashing": squash,
    }


def _splice_config(type_, name):
    return {"source": "predefined", "type": type_, "name": name}


def _make_checkpoint(rng, *, num_slots=1, num_tracks=3, usage=4, tissues=2, lora=False,
                     head_metadata=True, resolutions=(1,)):
    """Synthetic alphagenome_ft config.json + param tree (flat head keys, nested LoRA)."""
    o = num_slots
    params = {}
    for r in resolutions:
        dim = 1536 if r == 1 else 3072
        params[f"head/rna_seq/resolution_{r}/multi_organism_linear"] = {
            "w": rng.normal(size=(o, dim, num_tracks)).astype(np.float32),
            "b": rng.normal(size=(o, num_tracks)).astype(np.float32),
        }
        params[f"head/rna_seq/resolution_{r}"] = {
            "learnt_scale": rng.normal(size=(o, num_tracks)).astype(np.float32),
        }
    params["head/splice_site/multi_organism_linear"] = {
        "w": rng.normal(size=(o, 1536, 5)).astype(np.float32),
        "b": rng.normal(size=(o, 5)).astype(np.float32),
    }
    params["head/splice_usage/multi_organism_linear"] = {
        "w": rng.normal(size=(o, 1536, usage)).astype(np.float32),
        "b": rng.normal(size=(o, usage)).astype(np.float32),
    }
    params["head/splice_junctions/multi_organism_linear"] = {
        "w": rng.normal(size=(o, 1536, 768)).astype(np.float32),
        "b": rng.normal(size=(o, 768)).astype(np.float32),
    }
    for x in ("pos_donor", "pos_acceptor", "neg_donor", "neg_acceptor"):
        params[f"head/splice_junctions/{x}_logits"] = {
            "embeddings": rng.normal(size=(o, 2 * tissues * 768)).astype(np.float32),
        }
    if lora:
        tower = {}
        for i in range(9):
            block = "mha_block" if i == 0 else f"mha_block_{i}"
            tower[block] = {
                "q_layer": {"lora_a": rng.normal(size=(1536, RANK)).astype(np.float32),
                            "lora_b": rng.normal(size=(RANK, 1024)).astype(np.float32)},
                "v_layer": {"lora_a": rng.normal(size=(1536, RANK)).astype(np.float32),
                            "lora_b": rng.normal(size=(RANK, 192)).astype(np.float32)},
            }
        params["alphagenome"] = {"transformer_tower": tower}

    config = {
        "custom_heads": ["rna_seq", "splice_site", "splice_usage", "splice_junctions"],
        "head_configs": {
            "rna_seq": _track_head_config(resolutions=resolutions),
            "splice_site": _splice_config("SPLICE_SITES_CLASSIFICATION", "splice_site"),
            "splice_usage": _splice_config("SPLICE_SITES_USAGE", "splice_usage"),
            "splice_junctions": _splice_config("SPLICE_SITES_JUNCTION", "splice_junctions"),
        },
        "save_full_model": False,
        "save_minimal_model": False,
        "use_encoder_output": False,
        "backbone_lora": (
            {"rank": RANK, "alpha": ALPHA, "targets": ["q_proj", "v_proj"], "gradient_checkpointing": False}
            if lora else None
        ),
    }
    if head_metadata:
        organisms = ["HOMO_SAPIENS", "MUS_MUSCULUS"][:o]
        config["head_metadata"] = {
            "rna_seq": {
                org: [{"name": f"{org[:3]}_t{i}", "strand": "+", "nonzero_mean": 1.0 + i + 10 * k}
                      for i in range(num_tracks)]
                for k, org in enumerate(organisms)
            },
            "splice_site": {org: [{"name": n, "strand": "+"} for n in
                                  ["donor_pos", "acceptor_pos", "donor_neg", "acceptor_neg", "none"]]
                            for org in organisms},
            "splice_usage": {org: [{"name": f"s{i}", "strand": "+"} for i in range(usage)]
                             for org in organisms},
            "splice_junctions": {org: [{"name": f"tissue{i}", "strand": "+"} for i in range(tissues)]
                                 for org in organisms},
        }
    return config, params


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestHeadMapping:
    def test_track_head_mapping(self, rng):
        config, params = _make_checkpoint(rng, resolutions=(1, 128))
        conv = build_finetuned_conversion(config, params)
        sd = conv.state_dict
        for r in (1, 128):
            src = params[f"head/rna_seq/resolution_{r}/multi_organism_linear"]
            np.testing.assert_array_equal(sd[f"heads.rna_seq.convs.{r}.weight"].numpy(),
                                          src["w"].transpose(0, 2, 1))
            np.testing.assert_array_equal(sd[f"heads.rna_seq.convs.{r}.bias"].numpy(), src["b"])
            np.testing.assert_array_equal(sd[f"heads.rna_seq.residual_scales.{r}"].numpy(),
                                          params[f"head/rna_seq/resolution_{r}"]["learnt_scale"])
        assert sd["heads.rna_seq.convs.128.weight"].shape == (1, 3, 3072)
        np.testing.assert_array_equal(sd["heads.rna_seq.track_means"].numpy(), [[1.0, 2.0, 3.0]])
        head = conv.transfer_config.new_heads["rna_seq"]
        assert head["modality"] == "rna_seq" and head["num_tracks"] == 3
        assert head["resolutions"] == [1, 128]
        assert conv.transfer_config.mode == "linear"

    def test_splice_head_mapping(self, rng):
        config, params = _make_checkpoint(rng, usage=4, tissues=2)
        sd = build_finetuned_conversion(config, params).state_dict
        for pt, jx in [("splice_sites_classification_head", "splice_site"),
                       ("splice_sites_usage_head", "splice_usage"),
                       ("splice_sites_junction_head", "splice_junctions")]:
            src = params[f"head/{jx}/multi_organism_linear"]
            np.testing.assert_array_equal(sd[f"{pt}.conv.weight"].numpy(), src["w"].transpose(0, 2, 1))
            np.testing.assert_array_equal(sd[f"{pt}.conv.bias"].numpy(), src["b"])
        for x in ("pos_donor", "pos_acceptor", "neg_donor", "neg_acceptor"):
            got = sd[f"splice_sites_junction_head.rope_params.{x}"]
            assert got.shape == (1, 2, 2, 768)
            np.testing.assert_array_equal(
                got.numpy(), params[f"head/splice_junctions/{x}_logits"]["embeddings"].reshape(1, 2, 2, 768),
            )

    def test_lora_mapping(self, rng):
        config, params = _make_checkpoint(rng, lora=True)
        conv = build_finetuned_conversion(config, params)
        tc = conv.transfer_config
        assert tc.mode == "lora" and tc.lora_rank == RANK and tc.lora_alpha == ALPHA
        assert tc.lora_targets == ["q_proj", "v_proj"]
        tower = params["alphagenome"]["transformer_tower"]
        for i in (0, 5):
            block = tower["mha_block" if i == 0 else f"mha_block_{i}"]
            for proj in ("q", "v"):
                leaves = block[f"{proj}_layer"]
                prefix = f"tower.blocks.{i}.mha.{proj}_proj"
                np.testing.assert_array_equal(conv.state_dict[f"{prefix}.lora_A.weight"].numpy(), leaves["lora_a"].T)
                np.testing.assert_array_equal(conv.state_dict[f"{prefix}.lora_B.weight"].numpy(), leaves["lora_b"].T)
        assert sum(".lora_" in k for k in conv.state_dict) == 36

    def test_metadata_names_and_rows(self, rng):
        config, params = _make_checkpoint(rng, usage=2, tissues=1)
        meta = build_finetuned_conversion(config, params).metadata
        assert meta["organism"] == "human" and meta["organism_indices"] == [0]
        assert meta["track_names"]["rna_seq"] == ["HOM_t0", "HOM_t1", "HOM_t2"]
        assert meta["track_names"]["splice_usage"] == ["s0", "s1"]
        assert meta["track_names"]["splice_junctions"] == ["tissue0_pos", "tissue0_neg"]
        assert meta["track_names"]["splice_site"][0] == "cls_donor_pos"
        rna_rows = [r for r in meta["track_metadata"] if r["output_name"] == "rna_seq"]
        assert rna_rows[1] == {"strand": "+", "nonzero_mean": 2.0, "output_name": "rna_seq",
                               "track_index": 1, "track_name": "HOM_t1", "organism": 0}
        assert meta["resolutions"]["rna_seq"] == (1,)


class TestOrganismSelection:
    def test_multi_slot_requires_organism(self, rng):
        config, params = _make_checkpoint(rng, num_slots=2)
        with pytest.raises(ValueError, match="organism slots"):
            build_finetuned_conversion(config, params)

    @pytest.mark.parametrize("organism,slot,means", [("human", 0, [1.0, 2.0, 3.0]),
                                                     ("MUS_MUSCULUS", 1, [11.0, 12.0, 13.0])])
    def test_selects_slot(self, rng, organism, slot, means):
        config, params = _make_checkpoint(rng, num_slots=2)
        conv = build_finetuned_conversion(config, params, organism=organism)
        w = params["head/rna_seq/resolution_1/multi_organism_linear"]["w"]
        np.testing.assert_array_equal(conv.state_dict["heads.rna_seq.convs.1.weight"].numpy(),
                                      w[slot:slot + 1].transpose(0, 2, 1))
        emb = params["head/splice_junctions/pos_donor_logits"]["embeddings"]
        np.testing.assert_array_equal(conv.state_dict["splice_sites_junction_head.rope_params.pos_donor"].numpy(),
                                      emb[slot:slot + 1].reshape(1, 2, 2, 768))
        np.testing.assert_array_equal(conv.state_dict["heads.rna_seq.track_means"].numpy(), [means])
        assert conv.metadata["organism_indices"] == [slot]

    def test_single_slot_organism_mismatch(self, rng):
        config, params = _make_checkpoint(rng)
        with pytest.raises(ValueError, match="trained for HOMO_SAPIENS"):
            build_finetuned_conversion(config, params, organism="mouse")

    def test_single_slot_without_metadata_takes_requested(self, rng):
        config, params = _make_checkpoint(rng, head_metadata=False)
        conv = build_finetuned_conversion(config, params, organism="mouse", track_means=["rna_seq=1,2,3"])
        assert conv.metadata["organism"] == "mouse"


class TestTrackMeansFallback:
    def test_missing_means_error(self, rng):
        config, params = _make_checkpoint(rng, head_metadata=False)
        with pytest.raises(ValueError, match="no head_metadata.*--targets-config.*--track-means"):
            build_finetuned_conversion(config, params)

    def test_explicit_track_means(self, rng):
        config, params = _make_checkpoint(rng, head_metadata=False)
        conv = build_finetuned_conversion(config, params, track_means=["rna_seq=0.5,1.5,2.5"])
        np.testing.assert_allclose(conv.state_dict["heads.rna_seq.track_means"].numpy(), [[0.5, 1.5, 2.5]])
        assert conv.metadata["track_names"]["rna_seq"] == ["track_0", "track_1", "track_2"]
        assert conv.metadata["track_names"]["splice_junctions"] == [
            "tissue_0_pos", "tissue_1_pos", "tissue_0_neg", "tissue_1_neg",
        ]

    def test_explicit_track_means_wrong_count(self, rng):
        config, params = _make_checkpoint(rng, head_metadata=False)
        with pytest.raises(ValueError, match="expected 3"):
            build_finetuned_conversion(config, params, track_means={"rna_seq": [1.0]})

    def test_targets_config(self, rng, tmp_path):
        config, params = _make_checkpoint(rng, head_metadata=False)
        targets = tmp_path / "targets.yaml"
        targets.write_text(
            "heads:\n"
            "  - id: rna_seq\n    source: predefined\n    kind: rna_seq\n    targets:\n"
            "      - {path: a.bw, label: A, nonzero_mean: 0.25}\n"
            "      - {path: dir/b.bw, nonzero_mean: 4}\n"
            "      - {path: c.bw, label: C}\n"
        )
        conv = build_finetuned_conversion(config, params, targets_config=targets)
        np.testing.assert_allclose(conv.state_dict["heads.rna_seq.track_means"].numpy(), [[0.25, 4.0, 1.0]])
        assert conv.metadata["track_names"]["rna_seq"] == ["A", "b", "C"]

    def test_targets_config_without_means_errors(self, rng):
        config, params = _make_checkpoint(rng, head_metadata=False)
        targets = {"heads": [{"id": "rna_seq", "targets": [{"path": "a.bw"}] * 3}]}
        with pytest.raises(ValueError, match="No track means"):
            build_finetuned_conversion(config, params, targets_config=targets)

    def test_parse_track_means(self):
        assert parse_track_means(["a=1,2", "b=3"]) == {"a": [1.0, 2.0], "b": [3.0]}
        with pytest.raises(ValueError, match="HEAD=v1"):
            parse_track_means(["a"])


class TestUnsupported:
    def test_full_model(self, rng):
        config, params = _make_checkpoint(rng)
        config["save_full_model"] = True
        with pytest.raises(ValueError, match="full-model"):
            build_finetuned_conversion(config, params)

    def test_minimal_model(self, rng):
        config, params = _make_checkpoint(rng)
        config["save_minimal_model"] = True
        with pytest.raises(ValueError, match="minimal-model"):
            build_finetuned_conversion(config, params)

    def test_lora_other_target_in_config(self, rng):
        config, params = _make_checkpoint(rng, lora=True)
        config["backbone_lora"]["targets"] = ["q_proj", "k_proj"]
        with pytest.raises(ValueError, match="k_proj"):
            build_finetuned_conversion(config, params)

    def test_lora_leaves_on_other_layer(self, rng):
        config, params = _make_checkpoint(rng, lora=True)
        params["alphagenome"]["transformer_tower"]["mha_block"]["k_layer"] = {
            "lora_a": np.zeros((1536, RANK), np.float32), "lora_b": np.zeros((RANK, 128), np.float32),
        }
        with pytest.raises(ValueError, match="other than q/v"):
            build_finetuned_conversion(config, params)

    def test_backbone_params_present(self, rng):
        config, params = _make_checkpoint(rng)
        params["alphagenome/sequence_encoder/dna_embedder/conv1_d"] = {"w": np.zeros((15, 4, 768), np.float32)}
        with pytest.raises(ValueError, match="outside the converted heads"):
            build_finetuned_conversion(config, params)

    def test_custom_head(self, rng):
        config, params = _make_checkpoint(rng)
        config["head_configs"]["rna_seq"]["source"] = "custom"
        with pytest.raises(ValueError, match="custom"):
            build_finetuned_conversion(config, params)

    def test_squashing_mismatch(self, rng):
        config, params = _make_checkpoint(rng)
        config["head_configs"]["rna_seq"]["apply_squashing"] = False
        with pytest.raises(ValueError, match="apply_squashing"):
            build_finetuned_conversion(config, params)


def test_is_alphagenome_ft_checkpoint(tmp_path):
    assert not is_alphagenome_ft_checkpoint(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"dtype": "bfloat16"}))
    assert not is_alphagenome_ft_checkpoint(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"custom_heads": ["rna_seq"]}))
    assert is_alphagenome_ft_checkpoint(tmp_path)


def test_converted_delta_loads_with_load_finetuned_model(tmp_path, monkeypatch):
    """Orbax checkpoint dir -> .delta.pth -> load_finetuned_model reproduces the mapped weights."""
    ocp = pytest.importorskip("orbax.checkpoint")
    from alphagenome_pytorch.extensions.finetuning import transfer as tr_mod
    from alphagenome_pytorch.extensions.finetuning.checkpointing import load_finetuned_model

    config, params = _make_checkpoint(np.random.default_rng(1), lora=True, usage=2, tissues=1)
    ckpt_dir = tmp_path / "jax_ft"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text(json.dumps(config))
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(str(ckpt_dir / "checkpoint"), (params, {}))
    checkpointer.wait_until_finished()

    out = tmp_path / "converted.delta.pth"
    conversion = convert_finetuned_checkpoint(ckpt_dir, out)
    saved = torch.load(out, weights_only=False)
    assert saved["delta_checkpoint_version"] == 1
    assert saved["metadata"]["organism"] == "human"

    # Trunk weights are irrelevant here; skip reading a pretrained file.
    monkeypatch.setattr(tr_mod, "load_trunk", lambda model, *_, **__: model)
    model, meta = load_finetuned_model(out, "unused.safetensors", merge=False)
    assert meta["head_names"] == ["rna_seq", "splice_site", "splice_usage", "splice_junctions"]
    assert meta["track_names"] == conversion.metadata["track_names"]
    assert meta["default_organism_index"] == 0
    state = model.state_dict()
    for key, value in conversion.state_dict.items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0, msg=key)
    assert list(model.heads) == ["rna_seq"]
    assert model.splice_sites_junction_head is not None

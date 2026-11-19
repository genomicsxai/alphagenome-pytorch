"""
Unit tests for fine-tuning training utilities.

Tests the training module functions including collate, loss computation,
modality configuration, and utility functions.
"""

import pytest
import torch
from torch import Tensor

from alphagenome_pytorch.extensions.finetuning.training import (
    create_lr_scheduler,
    compute_finetuning_loss,
    collate_genomic,
    MODALITY_CONFIGS,
    ModalityConfig,
)

@pytest.mark.unit
class TestCollateGenomic:
    """Tests for collate_genomic function."""

    def test_collate_rnaseq_format(self):
        """Test collate with RNA-seq dict format."""
        # Simulate RNA-seq dataset output: (seq, {res: targets})
        batch = [
            (torch.randn(256, 4), {128: torch.randn(256, 3)}),
            (torch.randn(256, 4), {128: torch.randn(256, 3)}),
        ]

        sequences, targets_dict = collate_genomic(batch)

        assert sequences.shape == (2, 256, 4)
        assert 128 in targets_dict
        assert targets_dict[128].shape == (2, 256, 3)

    def test_collate_atac_format(self):
        """Test collate with ATAC tensor format."""
        # Simulate ATAC dataset output: (seq, {res: targets})
        batch = [
            (torch.randn(256, 4), {128: torch.randn(256, 5)}),
            (torch.randn(256, 4), {128: torch.randn(256, 5)}),
        ]

        sequences, targets_dict = collate_genomic(batch)

        assert sequences.shape == (2, 256, 4)
        assert 128 in targets_dict
        assert targets_dict[128].shape == (2, 256, 5)


@pytest.mark.unit
class TestCreateLrScheduler:
    """Tests for learning rate scheduler creation."""

    def test_scheduler_warmup(self):
        """Test scheduler produces warmup behavior."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        scheduler = create_lr_scheduler(optimizer, warmup_steps=100, total_steps=1000)

        # At step 0, LR should be 0
        assert scheduler.get_last_lr()[0] == 0.0

        # Step through warmup
        for _ in range(50):
            scheduler.step()

        # At step 50 (halfway through warmup), LR should be ~0.5
        assert 0.4e-3 < scheduler.get_last_lr()[0] < 0.6e-3

    def test_scheduler_decay(self):
        """Test scheduler produces cosine decay after warmup."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        scheduler = create_lr_scheduler(optimizer, warmup_steps=100, total_steps=1000)

        # Complete warmup
        for _ in range(100):
            scheduler.step()

        # At end of warmup, LR should be ~1.0
        lr_at_warmup_end = scheduler.get_last_lr()[0]
        assert 0.9e-3 < lr_at_warmup_end < 1.1e-3

        # Continue stepping
        for _ in range(450):  # step 100-550
            scheduler.step()

        # At halfway through decay, LR should be ~0.5
        lr_at_mid = scheduler.get_last_lr()[0]
        assert 0.4e-3 < lr_at_mid < 0.6e-3


@pytest.mark.unit
class TestComputeFinetuningLoss:
    """Tests for compute_finetuning_loss function."""

    def test_loss_single_resolution(self):
        """Test loss computation with single resolution (NLC format)."""
        # NLC format: (B, S, C) = (2, 5, 256) where S=5 seq positions, C=256 tracks
        predictions = {128: torch.randn(2, 5, 256).abs()}
        targets = {128: torch.randn(2, 5, 256).abs()}
        resolution_weights = {128: 1.0}
        device = torch.device("cpu")

        loss, loss_dict = compute_finetuning_loss(
            predictions=predictions,
            targets=targets,
            resolution_weights=resolution_weights,
            positional_weight=5.0,
            device=device,
            channels_last=True,
        )

        assert loss.ndim == 0  # Scalar
        assert "loss" in loss_dict
        assert "loss_128bp" in loss_dict

    def test_loss_dual_resolution(self):
        """Test loss computation with dual resolution (NLC format)."""
        predictions = {
            1: torch.randn(2, 3, 256).abs(),
            128: torch.randn(2, 3, 256).abs(),
        }
        targets = {
            1: torch.randn(2, 3, 256).abs(),
            128: torch.randn(2, 3, 256).abs(),
        }
        resolution_weights = {1: 1.0, 128: 1.0}
        device = torch.device("cpu")

        loss, loss_dict = compute_finetuning_loss(
            predictions=predictions,
            targets=targets,
            resolution_weights=resolution_weights,
            positional_weight=5.0,
            device=device,
            channels_last=True,
        )

        assert loss.ndim == 0  # Scalar
        assert "loss" in loss_dict
        assert "loss_1bp" in loss_dict
        assert "loss_128bp" in loss_dict

    def test_loss_ncl_format(self):
        """Test loss computation with NCL format (B, C, S)."""
        # NCL format: (B, C, S) = (2, 256, 5)
        predictions = {128: torch.randn(2, 256, 5).abs()}
        targets = {128: torch.randn(2, 256, 5).abs()}
        resolution_weights = {128: 1.0}
        device = torch.device("cpu")

        loss, loss_dict = compute_finetuning_loss(
            predictions=predictions,
            targets=targets,
            resolution_weights=resolution_weights,
            positional_weight=5.0,
            device=device,
            channels_last=False,
        )

        assert loss.ndim == 0  # Scalar
        assert "loss" in loss_dict
        assert "loss_128bp" in loss_dict

        # Verify it gives same result as transposed NLC
        predictions_nlc = {128: predictions[128].transpose(1, 2)}
        targets_nlc = {128: targets[128].transpose(1, 2)}
        loss_nlc, _ = compute_finetuning_loss(
            predictions=predictions_nlc,
            targets=targets_nlc,
            resolution_weights=resolution_weights,
            positional_weight=5.0,
            device=device,
            channels_last=True,
        )
        assert torch.isclose(loss, loss_nlc)

    def test_loss_weighted_resolutions(self):
        """Test loss computation respects resolution weights."""
        # NLC format: (B, S, C)
        predictions = {
            1: torch.randn(2, 3, 256).abs(),
            128: torch.randn(2, 3, 256).abs(),
        }
        targets = {
            1: torch.randn(2, 3, 256).abs(),
            128: torch.randn(2, 3, 256).abs(),
        }
        device = torch.device("cpu")

        # Equal weights
        loss_equal, _ = compute_finetuning_loss(
            predictions=predictions,
            targets=targets,
            resolution_weights={1: 1.0, 128: 1.0},
            positional_weight=5.0,
            device=device,
        )

        # Zero weight on 1bp
        loss_128_only, loss_dict = compute_finetuning_loss(
            predictions=predictions,
            targets=targets,
            resolution_weights={1: 0.0, 128: 1.0},
            positional_weight=5.0,
            device=device,
        )

        # They should differ (unless by chance 1bp loss is 0)
        # At minimum, the 128-only should be >= just the 128bp component
        assert loss_128_only.item() == pytest.approx(
            loss_dict["loss_128bp"].item(), rel=1e-5
        )

    def test_loss_skips_missing_resolutions(self):
        """Test loss computation skips resolutions not in predictions."""
        # NLC format: (B, S, C)
        predictions = {128: torch.randn(2, 3, 256).abs()}
        targets = {128: torch.randn(2, 3, 256).abs()}
        # Request both resolutions but only 128 is present
        resolution_weights = {1: 1.0, 128: 1.0}
        device = torch.device("cpu")

        loss, loss_dict = compute_finetuning_loss(
            predictions=predictions,
            targets=targets,
            resolution_weights=resolution_weights,
            positional_weight=5.0,
            device=device,
        )

        # Should only have 128bp loss
        assert "loss_128bp" in loss_dict
        assert "loss_1bp" not in loss_dict

    def test_dynamic_multinomial_resolution(self):
        """Test that multinomial_resolution is computed dynamically from seq_len."""
        # NLC format: (B, S, C)
        # Sequence length 256: multinomial_resolution should be 256 // 8 = 32
        predictions_256 = {128: torch.randn(2, 256, 3).abs()}
        targets_256 = {128: torch.randn(2, 256, 3).abs()}

        # Sequence length 64: multinomial_resolution should be 64 // 8 = 8
        predictions_64 = {128: torch.randn(2, 64, 3).abs()}
        targets_64 = {128: torch.randn(2, 64, 3).abs()}

        device = torch.device("cpu")

        # Both should work without errors (different multinomial_resolution internally)
        loss_256, _ = compute_finetuning_loss(
            predictions=predictions_256,
            targets=targets_256,
            resolution_weights={128: 1.0},
            positional_weight=5.0,
            device=device,
        )

        loss_64, _ = compute_finetuning_loss(
            predictions=predictions_64,
            targets=targets_64,
            resolution_weights={128: 1.0},
            positional_weight=5.0,
            device=device,
        )

        assert loss_256.ndim == 0
        assert loss_64.ndim == 0


@pytest.mark.unit
class TestModalityConfigs:
    """Tests for MODALITY_CONFIGS registry."""

    def test_all_modalities_present(self):
        """Test all expected modalities are defined."""
        expected = {'rna_seq', 'atac', 'dnase', 'procap', 'cage', 'chip_tf', 'chip_histone'}
        assert set(MODALITY_CONFIGS.keys()) == expected

    def test_all_configs_are_modality_config(self):
        """Test all configs are ModalityConfig instances."""
        for name, config in MODALITY_CONFIGS.items():
            assert isinstance(config, ModalityConfig), f"{name} is not ModalityConfig"

    def test_chip_modalities_128bp_only(self):
        """Test ChIP modalities only support 128bp resolution."""
        assert MODALITY_CONFIGS['chip_tf'].resolutions == (128,)
        assert MODALITY_CONFIGS['chip_histone'].resolutions == (128,)
        assert MODALITY_CONFIGS['chip_tf'].default_resolution_weights == {128: 1.0}
        assert MODALITY_CONFIGS['chip_histone'].default_resolution_weights == {128: 1.0}

    def test_dual_resolution_modalities(self):
        """Test modalities that support both 1bp and 128bp."""
        dual_res_types = ['rna_seq', 'atac', 'dnase', 'procap', 'cage']
        for mod_name in dual_res_types:
            config = MODALITY_CONFIGS[mod_name]
            assert config.resolutions == (1, 128), f"{mod_name} should have dual resolution"
            assert 1 in config.default_resolution_weights
            assert 128 in config.default_resolution_weights

    def test_all_modalities_have_embedding_dim(self):
        """Test all modalities have embedding_dim set."""
        for name, config in MODALITY_CONFIGS.items():
            assert config.embedding_dim == 3072, f"{name} should have embedding_dim=3072"

    def test_modality_config_name_matches_key(self):
        """Test that config.name matches the dict key."""
        for key, config in MODALITY_CONFIGS.items():
            assert config.name == key, f"Config name '{config.name}' doesn't match key '{key}'"


@pytest.mark.unit
@pytest.mark.parametrize(
    "requested, expected",
    [("cuda:2", [torch.device("cuda:2")]), ("mps", ["mps"]), ("cpu", [])],
)
def test_profile_synchronizes_selected_accelerator(monkeypatch, requested, expected):
    from alphagenome_pytorch.extensions.finetuning.training import _synchronize_device

    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", calls.append)
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("mps"))

    _synchronize_device(torch.device(requested))

    assert calls == expected


@pytest.mark.unit
class TestAmpContext:
    """The autocast the loops run under has to follow the model and the device.

    A hard-coded ``device_type="cuda"`` disabled itself on any other device
    while the model still cast its activations to bf16, so the first conv of
    the DNA embedder met fp32 weights and a CPU run died on batch one.
    """

    class _Policy:
        def __init__(self, compute_dtype):
            self.compute_dtype = compute_dtype

    class _Model(torch.nn.Module):
        def __init__(self, compute_dtype):
            super().__init__()
            self.dtype_policy = TestAmpContext._Policy(compute_dtype)

    def test_device_type_follows_the_device(self):
        from alphagenome_pytorch.extensions.finetuning.training import amp_context_for

        ctx = amp_context_for(self._Model(torch.bfloat16), torch.device("cpu"), True)
        assert ctx.device == "cpu"
        assert ctx._enabled is True

    def test_accepts_a_device_string(self):
        from alphagenome_pytorch.extensions.finetuning.training import amp_context_for

        assert amp_context_for(self._Model(torch.bfloat16), "cpu", True).device == "cpu"

    @pytest.mark.parametrize(
        "compute_dtype, use_amp",
        [(torch.float32, True), (torch.bfloat16, False)],
        ids=["float32-policy", "no-amp"],
    )
    def test_disables_ambient_autocast(self, compute_dtype, use_amp):
        from alphagenome_pytorch.extensions.finetuning.training import amp_context_for

        policy_model = self._Model(compute_dtype)
        layer = torch.nn.Linear(2, 2)
        sample = torch.ones(1, 2)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert layer(sample).dtype == torch.bfloat16
            with amp_context_for(policy_model, "cpu", use_amp):
                assert layer(sample).dtype == torch.float32

    def test_float32_skips_autocast_on_older_mps(self, monkeypatch):
        from alphagenome_pytorch.extensions.finetuning.training import amp_context_for

        def unavailable(*args, **kwargs):
            pytest.fail("A float32 policy must not construct MPS autocast on this build")

        monkeypatch.setattr(torch, "is_autocast_enabled", lambda: False)
        monkeypatch.setattr(torch, "autocast", unavailable)

        with amp_context_for(self._Model(torch.float32), "mps", True):
            pass

    @pytest.mark.parametrize("raises", [True, False], ids=["unsupported", "disabled"])
    def test_bfloat16_autocast_reports_unsupported_mps(self, monkeypatch, raises):
        from contextlib import nullcontext

        from alphagenome_pytorch.extensions.finetuning.training import (
            validate_bfloat16_autocast,
        )

        def unavailable(**kwargs):
            if raises:
                raise RuntimeError("unsupported autocast device")
            return nullcontext()

        monkeypatch.setattr(torch, "autocast", unavailable)
        monkeypatch.setattr(torch, "is_autocast_enabled", lambda device_type: False)

        with pytest.raises(ValueError, match=r"unavailable on mps.*--dtype float32"):
            validate_bfloat16_autocast(torch.device("mps"))

    def test_unwraps_compile_and_ddp_wrappers(self):
        """The loops see the wrapped model, so the policy lookup has to reach
        through both layers the runner adds (base -> DDP -> compile)."""
        from alphagenome_pytorch.extensions.finetuning.training import (
            model_compute_dtype,
        )

        base = self._Model(torch.float32)

        class _DDP(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

        class _Compiled(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self._orig_mod = module

        assert model_compute_dtype(_Compiled(_DDP(base))) is torch.float32

    def test_model_without_policy_does_not_enable_cpu_amp(self):
        from alphagenome_pytorch.extensions.finetuning.training import amp_context_for

        model = torch.nn.Linear(2, 2)
        sample = torch.ones(1, 2)
        with amp_context_for(model, "cpu", True):
            assert model(sample).dtype == torch.float32
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with amp_context_for(model, "cpu", True):
                assert model(sample).dtype == torch.bfloat16

    def test_model_without_policy_retains_cuda_bfloat16(self, monkeypatch):
        from alphagenome_pytorch.extensions.finetuning.training import amp_context_for

        monkeypatch.setattr(torch, "autocast", lambda **kwargs: kwargs)
        assert amp_context_for(torch.nn.Linear(2, 2), "cuda:0", True) == {
            "device_type": "cuda", "dtype": torch.bfloat16,
        }

    @pytest.mark.parametrize("multihead", [False, True], ids=["ddp", "multihead"])
    @pytest.mark.parametrize("use_amp", [False, True], ids=["fp32", "bf16"])
    def test_encoder_only_validation_respects_amp(self, multihead, use_amp):
        from alphagenome_pytorch.extensions.finetuning.training import (
            validate_ddp,
            validate_multihead,
        )

        expected_dtype = torch.bfloat16 if use_amp else torch.float32

        class Encoder(self._Model):
            def __init__(self):
                super().__init__(torch.bfloat16)
                self.conv = torch.nn.Conv1d(4, 2, kernel_size=1)

            def forward(self, sequences, organism_index, *, encoder_only):
                assert encoder_only
                features = self.conv(sequences.transpose(1, 2))
                assert features.dtype == expected_dtype
                return {"encoder_output": features}

        class EmptyHead(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return {}

        model = Encoder()
        options = dict(
            val_loader=[(torch.zeros(1, 8, 4), {})],
            device=torch.device("cpu"),
            positional_weight=1.0,
            count_weight=1.0,
            use_amp=use_amp,
            compute_pearson=False,
            encoder_only=True,
        )
        if multihead:
            validate_multihead(
                model, {}, modality_weights={}, resolution_weights={}, **options
            )
        else:
            validate_ddp(model, EmptyHead(), resolution_weights={128: 1.0}, **options)

    def test_no_hardcoded_cuda_autocast_remains(self):
        """Guard against reintroducing the literal in a new loop."""
        import inspect

        import alphagenome_pytorch.extensions.finetuning.training as training

        source = inspect.getsource(training)
        assert 'device_type="cuda"' not in source
        assert "device_type='cuda'" not in source

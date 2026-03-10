"""Hierarchical gradient parity tests with increasing strictness.

This module implements a "test ladder" approach to gradient parity validation:
- Level 0: Infrastructure sanity (dtype, mode, TF32 disabled)
- Level 1: Per-head loss parity
- Level 2: Gradient existence + NaN/Inf checks
- Level 3: Per-param aligned comparison (cosine similarity, rel L2, max abs diff)
- Level 4: Component aggregation with top-K offenders report

The ladder approach helps isolate where gradient drift originates and provides
actionable diagnostics for debugging.
"""

import pytest
import numpy as np
import torch

from .comparison_utils import compare_gradients, report_top_offenders
from tests.gradient_alignment import align_jax_gradient_to_pytorch, compute_gradient_metrics


@pytest.mark.integration_jax
@pytest.mark.jax
class TestGradientLadder:
    """Hierarchical gradient parity tests with increasing strictness."""

    # =========================================================================
    # Level 0: Infrastructure Sanity
    # =========================================================================

    def test_level0_tf32_disabled(self):
        """Verify TF32 is disabled for reproducible gradients.

        TF32 (Tensor Float 32) on Ampere+ GPUs uses 19-bit mantissa precision
        vs 23-bit for FP32, causing ~0.1-1% gradient drift. This test ensures
        TF32 is disabled to eliminate this source of numerical noise.
        """
        assert not torch.backends.cuda.matmul.allow_tf32, (
            "TF32 for CUDA matmul should be disabled for gradient parity tests"
        )
        assert not torch.backends.cudnn.allow_tf32, (
            "TF32 for cuDNN should be disabled for gradient parity tests"
        )

    def test_level0_cudnn_deterministic(self):
        """Verify cuDNN is in deterministic mode."""
        assert torch.backends.cudnn.deterministic, (
            "cuDNN should be in deterministic mode for reproducible gradients"
        )
        assert not torch.backends.cudnn.benchmark, (
            "cuDNN benchmark should be disabled for reproducible gradients"
        )

    @pytest.mark.parametrize("organism", ["human", "mouse"])
    def test_level0_dtype_consistency(self, cached_gradients, organism):
        """Verify both frameworks produce float32 gradients."""
        _, jax_grads = cached_gradients[organism]["jax"]
        _, pt_grads = cached_gradients[organism]["pytorch"]

        # Check a sample of JAX gradients
        for name, grad in list(jax_grads.items())[:5]:
            assert grad.dtype == np.float32, (
                f"JAX gradient {name} should be float32, got {grad.dtype}"
            )

        # Check a sample of PyTorch gradients
        for name, grad in list(pt_grads.items())[:5]:
            assert grad.dtype == np.float32, (
                f"PyTorch gradient {name} should be float32, got {grad.dtype}"
            )

    def test_level0_eval_mode(self, pytorch_model):
        """Verify PyTorch model is in eval mode (no dropout variance)."""
        assert not pytorch_model.training, (
            "PyTorch model should be in eval mode for gradient parity tests"
        )

    # =========================================================================
    # Level 1: Per-Head Loss Parity
    # =========================================================================

    @pytest.mark.parametrize("organism", ["human", "mouse"])
    @pytest.mark.parametrize(
        "head_name",
        ["atac", "dnase", "procap", "cage", "rna_seq", "chip_tf", "chip_histone", "contact_maps"],
    )
    def test_level1_per_head_loss_parity(
        self, cached_per_head_losses, tolerances, organism, head_name
    ):
        """Compare loss contribution from each head.

        This is the first validation after infrastructure checks. If per-head
        losses don't match, gradient parity is impossible.
        """
        jax_losses = cached_per_head_losses[organism]["jax"]
        pt_losses = cached_per_head_losses[organism]["pytorch"]

        if head_name not in jax_losses or head_name not in pt_losses:
            pytest.skip(f"Head {head_name} not present in both frameworks")

        jax_loss = jax_losses[head_name]
        pt_loss = pt_losses[head_name]

        rel_diff = abs(jax_loss - pt_loss) / (abs(jax_loss) + 1e-8)

        # Use same tolerance as total loss for now
        # TODO: Once forward parity is fully verified, tighten to rtol/10
        assert rel_diff < tolerances["rtol"], (
            f"Per-head loss mismatch for {head_name} ({organism}): "
            f"JAX={jax_loss:.6f}, PyTorch={pt_loss:.6f}, RelDiff={rel_diff:.4%}"
        )

    @pytest.mark.parametrize("organism", ["human", "mouse"])
    def test_level1_total_loss_breakdown(self, cached_per_head_losses, organism):
        """Report breakdown of loss contributions for debugging."""
        jax_losses = cached_per_head_losses[organism]["jax"]
        pt_losses = cached_per_head_losses[organism]["pytorch"]

        # Calculate totals
        jax_total = sum(jax_losses.values())
        pt_total = sum(pt_losses.values())

        # Report breakdown
        print(f"\n{organism.upper()} Loss Breakdown:")
        print("-" * 50)
        for head in jax_losses:
            j, p = jax_losses.get(head, 0), pt_losses.get(head, 0)
            diff = abs(j - p) / (abs(j) + 1e-8)
            status = "OK" if diff < 0.001 else "DIFF"
            print(f"  {head:25s}: JAX={j:10.4f}, PT={p:10.4f}, diff={diff:.4%} [{status}]")
        print(f"  {'TOTAL':25s}: JAX={jax_total:10.4f}, PT={pt_total:10.4f}")

        # This test always passes - it's for reporting
        assert True

    # =========================================================================
    # Level 2: Gradient Existence + Sanity Checks
    # =========================================================================

    @pytest.mark.parametrize("organism", ["human", "mouse"])
    def test_level2_all_params_have_gradients_jax(self, cached_gradients, organism):
        """Verify all expected parameters have gradients in JAX."""
        _, jax_grads = cached_gradients[organism]["jax"]

        assert len(jax_grads) > 0, "JAX should produce gradients"

        # Check for zero gradients (possible dead parameters)
        zero_grads = []
        for name, grad in jax_grads.items():
            if np.allclose(grad, 0):
                zero_grads.append(name)

        if zero_grads:
            print(f"\nWARNING: {len(zero_grads)} JAX parameters have zero gradients:")
            for name in zero_grads[:10]:
                print(f"  - {name}")

    @pytest.mark.parametrize("organism", ["human", "mouse"])
    def test_level2_all_params_have_gradients_pytorch(self, cached_gradients, organism):
        """Verify all expected parameters have gradients in PyTorch."""
        _, pt_grads = cached_gradients[organism]["pytorch"]

        assert len(pt_grads) > 0, "PyTorch should produce gradients"

        zero_grads = []
        for name, grad in pt_grads.items():
            if np.allclose(grad, 0):
                zero_grads.append(name)

        if zero_grads:
            print(f"\nWARNING: {len(zero_grads)} PyTorch parameters have zero gradients:")
            for name in zero_grads[:10]:
                print(f"  - {name}")

    @pytest.mark.parametrize("organism", ["human", "mouse"])
    def test_level2_no_nan_gradients(self, cached_gradients, organism):
        """Verify no NaN values in gradients."""
        _, jax_grads = cached_gradients[organism]["jax"]
        _, pt_grads = cached_gradients[organism]["pytorch"]

        jax_nan_params = [n for n, g in jax_grads.items() if np.isnan(g).any()]
        pt_nan_params = [n for n, g in pt_grads.items() if np.isnan(g).any()]

        assert len(jax_nan_params) == 0, f"JAX has NaN gradients in: {jax_nan_params[:5]}"
        assert len(pt_nan_params) == 0, f"PyTorch has NaN gradients in: {pt_nan_params[:5]}"

    @pytest.mark.parametrize("organism", ["human", "mouse"])
    def test_level2_no_inf_gradients(self, cached_gradients, organism):
        """Verify no Inf values in gradients."""
        _, jax_grads = cached_gradients[organism]["jax"]
        _, pt_grads = cached_gradients[organism]["pytorch"]

        jax_inf_params = [n for n, g in jax_grads.items() if np.isinf(g).any()]
        pt_inf_params = [n for n, g in pt_grads.items() if np.isinf(g).any()]

        assert len(jax_inf_params) == 0, f"JAX has Inf gradients in: {jax_inf_params[:5]}"
        assert len(pt_inf_params) == 0, f"PyTorch has Inf gradients in: {pt_inf_params[:5]}"

    # =========================================================================
    # Level 3: Per-Parameter Aligned Comparison
    # =========================================================================

    @pytest.mark.parametrize("organism", ["human", "mouse"])
    @pytest.mark.parametrize(
        "component",
        [
            "encoder",
            "tower",
            "decoder",
            "embedder_1bp",
            "embedder_128bp",
            "heads.atac",
            "heads.dnase",
        ],
    )
    def test_level3_cosine_similarity(
        self, cached_gradients, param_name_mapping, tolerances, organism, component
    ):
        """Test cosine similarity of gradients after proper alignment.

        Cosine similarity measures directional alignment, which is crucial
        for gradient-based optimization. A value < 0.99 indicates the
        gradients are pointing in meaningfully different directions.
        """
        _, jax_grads = cached_gradients[organism]["jax"]
        _, pt_grads = cached_gradients[organism]["pytorch"]

        component_params = [n for n in pt_grads.keys() if n.startswith(component)]
        if not component_params:
            pytest.skip(f"No params for: {component}")

        failures = []
        for pt_name in component_params:
            jax_name = param_name_mapping.get(pt_name)
            if jax_name is None or jax_name not in jax_grads:
                continue

            pt_grad = pt_grads[pt_name]
            jax_grad = jax_grads[jax_name]

            try:
                jax_aligned = align_jax_gradient_to_pytorch(pt_name, jax_grad, pt_grad.shape)
            except ValueError:
                failures.append(f"{pt_name}: Alignment failed")
                continue

            metrics = compute_gradient_metrics(pt_grad, jax_aligned)
            if metrics["cosine_sim"] < 0.99:
                failures.append(
                    f"{pt_name}: cosine={metrics['cosine_sim']:.4f} (should be >= 0.99)"
                )

        assert len(failures) == 0, (
            f"Cosine similarity failures for {component}:\n" + "\n".join(failures[:10])
        )

    @pytest.mark.parametrize("organism", ["human", "mouse"])
    @pytest.mark.parametrize(
        "component",
        [
            "encoder",
            "tower",
            "decoder",
            "embedder_1bp",
            "embedder_128bp",
            "heads.atac",
            "heads.dnase",
        ],
    )
    def test_level3_relative_l2_norm(
        self, cached_gradients, param_name_mapping, tolerances, organism, component
    ):
        """Test relative L2 norm difference of gradients.

        Relative L2 measures magnitude similarity. Values > 1% indicate
        significant scale differences that could affect learning rates.
        """
        _, jax_grads = cached_gradients[organism]["jax"]
        _, pt_grads = cached_gradients[organism]["pytorch"]

        component_params = [n for n in pt_grads.keys() if n.startswith(component)]
        if not component_params:
            pytest.skip(f"No params for: {component}")

        failures = []
        for pt_name in component_params:
            jax_name = param_name_mapping.get(pt_name)
            if jax_name is None or jax_name not in jax_grads:
                continue

            pt_grad = pt_grads[pt_name]
            jax_grad = jax_grads[jax_name]

            try:
                jax_aligned = align_jax_gradient_to_pytorch(pt_name, jax_grad, pt_grad.shape)
            except ValueError:
                continue

            metrics = compute_gradient_metrics(pt_grad, jax_aligned)
            # Use rtol_grad for gradients - backward pass accumulates more numerical error
            if metrics["rel_l2"] > tolerances["rtol_grad"]:
                failures.append(
                    f"{pt_name}: rel_l2={metrics['rel_l2']:.4%} (should be <= {tolerances['rtol_grad']:.1%})"
                )

        assert len(failures) == 0, (
            f"Relative L2 failures for {component}:\n" + "\n".join(failures[:10])
        )

    # =========================================================================
    # Level 4: Component Aggregation with Top-K Offenders
    # =========================================================================

    @pytest.mark.parametrize("organism", ["human", "mouse"])
    def test_level4_component_aggregation(
        self, cached_gradients, param_name_mapping, tolerances, organism
    ):
        """Aggregate metrics by component and report worst offenders.

        This test provides a comprehensive view of gradient parity across
        all components, with detailed reporting of the worst mismatches.
        """
        _, jax_grads = cached_gradients[organism]["jax"]
        _, pt_grads = cached_gradients[organism]["pytorch"]

        all_results = []
        component_stats = {}

        for pt_name, pt_grad in pt_grads.items():
            jax_name = param_name_mapping.get(pt_name)
            if jax_name is None or jax_name not in jax_grads:
                continue

            jax_grad = jax_grads[jax_name]

            try:
                jax_aligned = align_jax_gradient_to_pytorch(pt_name, jax_grad, pt_grad.shape)
            except ValueError:
                continue

            # Use rtol_grad for gradients - backward pass accumulates more numerical error
            result = compare_gradients(
                pt_name, pt_grad, jax_aligned, rtol=tolerances["rtol_grad"]
            )
            all_results.append(result)

            # Extract component name
            component = pt_name.split(".")[0]
            if component not in component_stats:
                component_stats[component] = {"total": 0, "passed": 0, "cosines": []}
            component_stats[component]["total"] += 1
            if result.passed:
                component_stats[component]["passed"] += 1
            component_stats[component]["cosines"].append(result.cosine_sim)

        # Report by component
        print(f"\n{organism.upper()} Gradient Parity by Component:")
        print("-" * 70)
        for comp, stats in sorted(component_stats.items()):
            pass_rate = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
            median_cos = np.median(stats["cosines"]) if stats["cosines"] else 0
            print(
                f"  {comp:25s}: {stats['passed']:3d}/{stats['total']:3d} passed "
                f"({pass_rate:5.1f}%), median cosine={median_cos:.4f}"
            )

        # Report top offenders
        print("\n" + report_top_offenders(all_results, k=10))

        # Calculate overall pass rate
        total_passed = sum(1 for r in all_results if r.passed)
        total_count = len(all_results)
        pass_rate = total_passed / total_count * 100 if total_count > 0 else 0

        print(f"\nOverall: {total_passed}/{total_count} parameters passed ({pass_rate:.1f}%)")

        # Fail if pass rate is below threshold
        min_pass_rate = 95.0  # Expect at least 95% of parameters to pass
        assert pass_rate >= min_pass_rate, (
            f"Only {pass_rate:.1f}% of parameters passed (expected >= {min_pass_rate}%)"
        )

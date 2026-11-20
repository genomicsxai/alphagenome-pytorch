"""Normalized junction loss must distinguish zero targets from masked pairs."""
import math
import unittest

import torch

from alphagenome_pytorch.extensions.finetuning.training import _compute_junction_loss


def objective(pred, target, positions, junction_loss="normalized"):
    return _compute_junction_loss(
        pred[..., :2], target[..., :2], positions[:, 0], positions[:, 1],
        pred[..., 2:], target[..., 2:], positions[:, 2], positions[:, 3],
        pred.device, junction_loss=junction_loss,
    )


class TestEmptyJunctionLoss(unittest.TestCase):
    def test_zero_targets_still_penalize_positive_counts(self):
        pred = torch.full((1, 4, 4, 4), .5, requires_grad=True)
        target = torch.zeros_like(pred)
        positions = torch.arange(4).expand(1, 4, 4)
        loss = objective(pred, target, positions)
        # Uniform smoothed labels: two log(K) CEs, plus two Poisson totals.
        self.assertAlmostEqual(loss.item(), .4*math.log(4) + .08*2, places=6)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.full_like(pred, .005), atol=1e-7, rtol=1e-6)

    def test_no_pairs_has_connected_zero_gradient(self):
        pred = torch.ones(1, 4, 4, 4, requires_grad=True)
        loss = objective(pred, torch.zeros_like(pred), torch.full((1, 4, 4), -1))
        self.assertEqual(loss.item(), 0)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.zeros_like(pred))

    def test_empty_example_is_not_batch_composition_dependent(self):
        pred = torch.full((2, 4, 4, 4), .5, requires_grad=True)
        target = torch.ones_like(pred)
        target[0] = 0
        positions = torch.arange(4).expand(2, 4, 4)
        joint = objective(pred, target, positions)
        separate = sum(objective(pred[i:i+1], target[i:i+1], positions[i:i+1]) for i in range(2)) / 2
        torch.testing.assert_close(joint, separate)

    def test_all_masked_nonfinite_values_do_not_pollute_loss(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                pred = torch.full((1, 4, 4, 4), value, requires_grad=True)
                loss = objective(pred, torch.full_like(pred, value), torch.full((1, 4, 4), -1))
                self.assertEqual(loss.item(), 0)
                self.assertTrue(loss.requires_grad)
                loss.backward()
                torch.testing.assert_close(pred.grad, torch.zeros_like(pred))

    def test_legacy_modes_keep_zero_target_behavior(self):
        for mode in ("original", "sparse"):
            with self.subTest(mode=mode):
                pred = torch.full((1, 4, 4, 4), .5, requires_grad=True)
                loss = objective(pred, torch.zeros_like(pred), torch.arange(4).expand(1, 4, 4), mode)
                self.assertEqual(loss.item(), 0)
                self.assertFalse(loss.requires_grad)

    def test_legacy_modes_keep_all_masked_behavior(self):
        for mode in ("original", "sparse"):
            with self.subTest(mode=mode):
                pred = torch.ones(1, 4, 4, 4, requires_grad=True)
                loss = objective(pred, torch.ones_like(pred), torch.full((1, 4, 4), -1), mode)
                self.assertEqual(loss.item(), 0)
                self.assertFalse(loss.requires_grad)


if __name__ == '__main__':
    unittest.main(verbosity=2)

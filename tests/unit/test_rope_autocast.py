"""Regression coverage for sparse junction RoPE under CUDA autocast.

Can also run directly with Python/unittest, without the optional pytest package.
"""
import unittest

import torch

from alphagenome_pytorch.attention import apply_rope


class TestRopeAutocast(unittest.TestCase):
    def check_device(self, device):
        torch.manual_seed(1234)
        positions = torch.tensor(
            [[-1, 0, 1, 255, 256, 4095, 524287, 1048575],
             [1048575, 524289, 4096, 257, 256, 1, 0, -1]],
            device=device,
        )
        for dtype in (torch.float32, torch.bfloat16):
            for inplace in (False, True):
                with self.subTest(device=device, dtype=dtype, inplace=inplace):
                    x = torch.randn(2, 8, 2, 768, device=device).to(dtype)
                    weight = torch.randn_like(x)
                    results = []
                    for enabled in (False, True):
                        leaf = x.clone().requires_grad_(True)
                        with torch.autocast(device, dtype=torch.bfloat16, enabled=enabled):
                            result = apply_rope(leaf.clone(), positions, max_position=2**20, inplace=inplace)
                            loss = (result * weight).float().sum()
                        grad, = torch.autograd.grad(loss, leaf)
                        self.assertEqual(result.dtype, dtype)
                        self.assertTrue(torch.isfinite(result).all())
                        self.assertTrue(torch.isfinite(grad).all())
                        results.append((result.detach(), grad))
                    for ref, actual in zip(*results):
                        torch.testing.assert_close(actual, ref, rtol=1e-6, atol=1e-6)

    def test_cpu_forward_backward(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires CUDA autocast')
    def test_cuda_forward_backward(self):
        self.check_device('cuda')


if __name__ == '__main__':
    unittest.main(verbosity=2)

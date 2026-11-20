"""Regression for BF16 frequencies observed in the official JAX JIT reference.

These exact BF16 constants come from n_freq=64 and max_position=8192.
Rounding the denominator *before* the reciprocal gives different constants.
"""
import unittest
import torch
from alphagenome_pytorch.attention import apply_rope


class TestCompiledRopeFrequencies(unittest.TestCase):
    def check_device(self, device):
        positions = torch.tensor([[0, 1, 257, 2049, 8191]], device=device)
        x = torch.zeros(1, 5, 1, 128, device=device, dtype=torch.bfloat16)
        x[..., ::2] = 1
        actual = apply_rope(x, positions, max_position=8192)
        compiled_frequencies = {1: .46484375, 5: .1416015625, 7: .10302734375, 11: .06298828125}
        for index, frequency in compiled_frequencies.items():
            angle = positions.to(x.dtype) * torch.tensor(frequency, dtype=x.dtype, device=device)
            torch.testing.assert_close(actual[:, :, 0, 2*index], angle.cos(), rtol=0, atol=0)
            torch.testing.assert_close(actual[:, :, 0, 2*index+1], angle.sin(), rtol=0, atol=0)

    def test_cpu_compiled_frequencies(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires CUDA')
    def test_cuda_compiled_frequencies(self):
        self.check_device('cuda')


if __name__ == '__main__':
    unittest.main(verbosity=2)

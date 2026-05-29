import unittest

import torch
from amd.dsl import Math
import flydsl.compiler as flyc


class MathTest(unittest.TestCase):
    def test_cuda_pre(self):
        self.assertTrue(torch.cuda.is_available())

    def test_vector_add(self):
        n: int = 128
        A = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
        B = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
        C = torch.zeros(n, dtype=torch.float32).cuda()
        tA = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=0, divisibility=4)
        Math.vec_add(tA, B, C, n, n + 1, stream=torch.cuda.Stream())  # pyright: ignore[reportArgumentType]
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(C, A + B))


if __name__ == "__main__":
    unittest.main()

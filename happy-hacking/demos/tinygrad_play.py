from tinygrad import nn, device
from tinygrad.tensor import Tensor
from tinygrad.nn.datasets import mnist

"""
Built wit 4 layers:
1. pytorch like frontend.
2. a scheduler which breaks the compute into kernels.
3. a lowering engine which converts ASTs into code that can run on the accelerator.
4. an execution engine which can run the code.
"""


print(device.Device)


class Model:
    def __init__(self):
        self.l1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.l3 = nn.Linear(1600, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x).relu().max_pool2d((2, 2))  # type:ignore
        x = self.l2(x).relu().max_pool2d((2, 2))  # type:ignore
        return self.l3(x.flatten(1).dropout(0.5))


x_train, y_train, x_test, y_test = mnist()

model = Model()
acc = (model(x_test).argmax(axis=1) == y_test).mean()
print(acc.item())

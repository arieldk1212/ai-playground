import timeit
import pathlib

from tinygrad.nn import optim
from tinygrad.nn import state
from tinygrad import nn, device
from tinygrad.tensor import Tensor
from tinygrad.nn.datasets import mnist
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import GlobalCounters, Context

"""
Built wit 4 layers:
1. pytorch like frontend.
2. a scheduler which breaks the compute into kernels.
3. a lowering engine which converts ASTs into code that can run on the accelerator.
4. an execution engine which can run the code.

Frontend:
    * All the stuff in tensor.py is a syntactic sugar around constructing a graph of "UOps".
    * 2 Types of UOps: Base and View, base contains compute into a contiguous buffer,
        view is a view, inputs to base can be base or view, inputs to view can be only
        base.    

Scheduling:
    * Converts the graph of UOps into a LINEAR UOps whose src is a list of CALL UOps.
    * One CALL is one kernel on the GPU, and the scheduler is responsible for breaking the
        large computre graph into subgraphs that can fit in a kernel.

Lowering:
    * The code in engine/realize.py basically lowers each CALL  by compiling its AST into
        a program and running it. 
    * To get more info and deep dive into the complexity, start by examining the codegen folder.
    * First we lower the AST to UOps (Linear list to run), this is where the BEAM search
        happens, Then we render the UOps into code with a "Renderer", then we compile   the code
        to binary with a Compiler.

Execution:
    * run_linear walks the LINEAR UOp, dispatching each CALL to a runner (kernel, view, copy, encdec or graph).

Runtimes:
    * They handle device specific interactions, They handle tasks such as initializing devices, allocating memory,
        loading/launching programs and more.
"""

"""
Graph flow down to programs:

1. tinygrad/schedule -> Group UOps into kernels -- get_kernel_graph().
2. tinygrad/codegen/opt -> Transform the ast into an optimized ast. This is where BEAM search and heuristics live.
3. tinygrad/codegen -> Transform the optimized ast into a linearized and rendered program -- to_program().
4. tinygrad/renderer -> Transform the linearized list of UOps into a program, represented as a string -- render().
5. tinygrad/engine -> Abstracted high level interface to the runtimes -- to_program().
"""


class DeviceContainer:
    def __init__(self):
        self.device = device.Device.DEFAULT

    def __call__(self) -> str:
        print(self.device)
        return self.device

    def __repr__(self):
        return "DEFAULT DEVICE: " + self.device

    @property
    def get_default_device(self) -> str:
        return self.device


p_device: DeviceContainer = DeviceContainer()


# gguf_tensor = Tensor(pathlib.Path("Meta-Llama-3-8B-Instruct.Q4_0.gguf")).to(
#     p_device.get_default_device
# )
# kv_data, state_dict = gguf_load(gguf_tensor)


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

# Train
optim = optim.Adam(state.get_parameters(model))
batch_size: int = 128


def step() -> Tensor:
    Tensor.training = True  # Dropout!
    samples = Tensor.randint(batch_size, high=x_train.shape[0])
    x, y = x_train[samples], y_train[samples]
    optim.zero_grad()
    loss = model(x).sparse_categorical_crossentropy(y).backward()
    optim.step()
    return loss


tiny_step = TinyJit(step)


@TinyJit
def jit_step():
    Tensor.training = True  # Dropout!
    samples = Tensor.randint(batch_size, high=x_train.shape[0])
    x, y = x_train[samples], y_train[samples]
    optim.zero_grad()
    loss = model(x).sparse_categorical_crossentropy(y).backward()
    optim.step()
    return loss


# print("Step func")
# print(timeit.repeat(step, repeat=5, number=1))
# print("Tiny step func")
# print(timeit.repeat(tiny_step, repeat=5, number=1))


class Runtime:
    @staticmethod
    def run_with_context(func) -> None:
        GlobalCounters.reset()
        with Context(DEBUG=2):
            func()

    @staticmethod
    def train_debug(func):
        for st in range(7000):
            loss = func()
            if st % 100 == 0:
                Tensor.training = False
                acc = (model(x_test).argmax(axis=1) == y_test).mean().item()
                print(f"step {st:4d}, loss {loss.item():.2f}, acc {acc * 100.0:.2f}%")


Runtime.run_with_context(step)
# Runtime.run_with_context(tiny_step)

# Runtime.train_debug(step)
# Runtime.train_debug(jit_step)
# Runtime.train_debug(tiny_step)

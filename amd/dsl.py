import flydsl.expr as fx
import flydsl.compiler as flyc

# AMD Torch guide
# https://github.com/nikos230/Run-Pytorch-with-AMD-Radeon-GPU/blob/main/check_pytorch.py


class KernalMath:
    @staticmethod
    @flyc.kernel
    def vec_add_kernel(
        A: fx.Tensor,  # type: ignore
        B: fx.Tensor,  # type: ignore
        C: fx.Tensor,  # type:ignore
        block_dim: fx.Constexpr,
    ):
        bid: fx.Int32 = fx.block_idx.x
        tid: fx.Int32 = fx.thread_idx.x

        # In gpu we need to access the current block and get the core thread
        A = fx.rocdl.make_buffer_tensor(A)

        tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
        tB = fx.logical_divide(B, fx.make_layout(block_dim, 1))
        tC = fx.logical_divide(C, fx.make_layout(block_dim, 1))

        tA = fx.slice(tA, (None, bid))
        tB = fx.slice(tB, (None, bid))
        tC = fx.slice(tC, (None, bid))
        tA = fx.logical_divide(tA, fx.make_layout(1, 1))
        tB = fx.logical_divide(tB, fx.make_layout(1, 1))
        tC = fx.logical_divide(tC, fx.make_layout(1, 1))

        rab_mem_ref_ty = fx.MemRefType.get(
            fx.T.f32(), fx.LayoutType.get(1, 1), fx.AddressSpace.Register
        )

        copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        copy_atom_buffer = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

        rA = fx.memref_alloca(rab_mem_ref_ty, fx.make_layout(1, 1))
        rB = fx.memref_alloca(rab_mem_ref_ty, fx.make_layout(1, 1))
        rC = fx.memref_alloca(rab_mem_ref_ty, fx.make_layout(1, 1))

        fx.copy_atom_call(copy_atom_buffer, fx.slice(tA, (None, tid)), rA)
        fx.copy_atom_call(copy_atom, fx.slice(tB, (None, tid)), rB)

        vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
        fx.memref_store_vec(vC, rC)
        fx.copy_atom_call(copy_atom, rC, fx.slice(tC, (None, tid)))


class Math:
    @staticmethod
    def vec_add(
        A: fx.Tensor,  # type:ignore
        B: fx.Tensor,  # type:ignore
        C: fx.Tensor,  # type:ignore
        n: fx.Int32,
        const_n: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        block_dim: int = 64
        grid_x = (n + block_dim - 1) // block_dim

        KernalMath.vec_add_kernel(A, B, C, block_dim).launch(
            grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream.value
        )

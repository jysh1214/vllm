Output code: 
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cpp_fused__to_copy_add_mean_mul_pow_rsqrt_0 = async_compile.cpp_pybinding(['const at::BFloat16*', 'const at::BFloat16*', 'float*', 'at::BFloat16*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const at::BFloat16* in_ptr0,
                       const at::BFloat16* in_ptr1,
                       float* out_ptr0,
                       at::BFloat16* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::VectorizedN<float,2> tmp_acc0_vec = at::vec::VectorizedN<float,2>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(896L); x1+=static_cast<int64_t>(16L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(896L)))
                        {
                            auto tmp0 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 896L*x0), static_cast<int64_t>(16));
                            auto tmp1 = at::vec::convert<float,2,at::BFloat16,1>(tmp0);
                            auto tmp2 = tmp1 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 2>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(896L); x1+=static_cast<int64_t>(16L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(896L)))
                    {
                        auto tmp0 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 896L*x0), static_cast<int64_t>(16));
                        auto tmp2 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp10 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(16));
                        auto tmp1 = at::vec::convert<float,2,at::BFloat16,1>(tmp0);
                        auto tmp3 = static_cast<float>(896.0);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = static_cast<float>(1e-06);
                        auto tmp6 = float(tmp4 + tmp5);
                        auto tmp7 = 1 / std::sqrt(tmp6);
                        auto tmp8 = at::vec::VectorizedN<float,2>(tmp7);
                        auto tmp9 = tmp1 * tmp8;
                        auto tmp11 = at::vec::convert<float,2,at::BFloat16,1>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp13 = at::vec::convert<at::BFloat16,1,float,2>(tmp12);
                        tmp13.store(out_ptr1 + static_cast<int64_t>(x1 + 896L*x0), static_cast<int64_t>(16));
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1 = args
        args.clear()
        assert_size_stride(arg0_1, (8, 896), (896, 1))
        assert_size_stride(arg1_1, (896, ), (1, ))
        buf0 = empty_strided_cpu((8, 1), (1, 8), torch.float32)
        buf1 = empty_strided_cpu((8, 896), (896, 1), torch.bfloat16)
        cpp_fused__to_copy_add_mean_mul_pow_rsqrt_0(arg0_1, arg1_1, buf0, buf1)
        del arg0_1
        del arg1_1
        return (buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8, 896), (896, 1), device='cpu', dtype=torch.bfloat16)
    arg1_1 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.bfloat16)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))

Output code written to: /tmp/claude-1000/-home-alex-vllm/0c49407e-35f0-4e19-934a-cc4bf66ed4c1/scratchpad/run_cpu/cpp_kernel_cache/ch/cch24g2a4mh5yjowuvs63jxdweg6rwxoaed3fmwo2zrpolvgjphp.py
Output code written to: /tmp/claude-1000/-home-alex-vllm/0c49407e-35f0-4e19-934a-cc4bf66ed4c1/scratchpad/run_cpu/cpp_kernel_cache/ch/cch24g2a4mh5yjowuvs63jxdweg6rwxoaed3fmwo2zrpolvgjphp.py

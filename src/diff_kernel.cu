#include <cuda_runtime.h>

__global__
void diff_kernel(const uint64_t* __restrict__ prev,
                 const uint64_t* __restrict__ curr,
                 uint8_t*       __restrict__ flag,
                 size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    while (idx < N)
    {
        uint64_t p = __ldg(prev + idx);
        uint64_t c = __ldg(curr + idx);

        flag[idx] = (p == c) ? 0u : 1u;
        idx += stride;
    }
}

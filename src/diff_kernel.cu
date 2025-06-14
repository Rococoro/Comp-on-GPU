#include "diff_kernel.cuh"

__global__
void diff_kernel(const uint64_t* __restrict__ d_prev,
    const uint64_t* __restrict__ d_curr,
    uint8_t* __restrict__ d_flag,
    size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    d_flag[i] = (d_prev[i] == d_curr[i]) ? 0 : 1;
}

#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

__global__
void diff_kernel(const uint64_t* __restrict__ d_prev,
    const uint64_t* __restrict__ d_curr,
    uint8_t* __restrict__ d_flag,
    size_t N);

inline void launch_diff(const uint64_t* __restrict__ d_prev,
    const uint64_t* __restrict__ d_curr,
    uint8_t* __restrict__ d_flag,
    size_t N,
    cudaStream_t stream = 0)
{
    const int THREADS = 256;
    dim3 block(THREADS);
    dim3 grid((N + THREADS - 1) / THREADS);
    diff_kernel<<<grid, block, 0, stream>>>(d_prev, d_curr, d_flag, N);
}
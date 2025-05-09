#include "gather_kernel.cuh"

/* 256‑thread 블록 기준: 1 스레드 - 1 변경 블록 */
__global__
void gather_ptrs(const int32_t* __restrict__ deltaIdx, const char* __restrict__ base, std::size_t blkSz, char** __restrict__ outPtrs, std::size_t M)
{
    std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= M) return;

    int32_t blkID = deltaIdx[k];               // 변경 블록 번호
    outPtrs[k]    = const_cast<char*>(base)    // 원본 버퍼 시작 +
                  + static_cast<std::size_t>(blkID) * blkSz;
}
#include "gather_kernel.cuh"

//1 스레드 - 1 변경된 블록
__global__
void gather_ptrs(const int32_t* __restrict__ deltaIdx, const char* __restrict__ base, std::size_t blkSz, char** __restrict__ outPtrs, std::size_t M)
{
    std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ char* s_ptr[];
    // 주소 계산
    char* addr = nullptr; //if문 밖의 thread는 null
    if (k < M) {
        int32_t blkID = deltaIdx[k];
        addr = const_cast<char*>(base) + static_cast<std::size_t>(blkID) * blkSz;
    }
    s_ptr[threadIdx.x] = addr;

    __syncthreads(); // 모든 주소 계산 완료에 대한 coalescing
    if (k < M) // 범위 안 스레드만 실제 결과 저장
        outPtrs[k] = s_ptr[threadIdx.x];
}

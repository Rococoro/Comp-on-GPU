
#include "MurmurHash2.cuh"

__device__ __forceinline__
uint64_t MurmurHash64A_GPU(const void* key, int len, uint64_t seed = 0x1e35a7bdUL)
{
    return MurmurHash64A(key, len, seed);
}

//d_in: 원본 바이트 버퍼, d_out: 해쉬 결과 배열(device)
//blk_sz, N: 블락 사이즈, 개수
__global__ void hash_kernel(const char* d_in, uint64_t* d_out, size_t blk_sz, size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        d_out[i] = MurmurHash64A_GPU(d_in + i*blk_sz, blk_sz);
}
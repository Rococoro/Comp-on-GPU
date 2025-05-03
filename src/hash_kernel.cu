
#include "MurmurHash2.h"

__device__ __forceinline__
uint64_t MurmurHash64A_GPU(const void* key, int len, uint64_t seed = 0x1e35a7bdUL)
{
    return MurmurHash64A(key, len, seed);
}
#pragma once
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace dz
{

/**
 * deltaFilter
 *   @param d_flag  : device uint8_t* (0/1 플래그)  [size = N]
 *   @param N       : 블록 개수
 *   @param outIdx  : 결과 인덱스들이 push_back 될 device_vector<int>
 *   @param stream  : CUDA 스트림 (default 0)
 *   @return        : 변경 블록 개수
 */
    std::size_t deltaFilter(const uint8_t* d_flag, std::size_t N, thrust::device_vector<int>& outIdx, cudaStream_t stream = 0);

}
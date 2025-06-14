#pragma once
#include <cuda_runtime.h>
#include <cstddef>  // size_t
#include <cstdint>  // int32_t

/**
 * gather_ptrs ─ 변경 블록 번호 → 블록 시작 주소
 *
 *  @param deltaIdx  : 변경 블록 인덱스 배열  [device int32_t*]  (size = M)
 *  @param base      : 시계열 원본 버퍼 첫 주소 [device const char*]
 *  @param blkSz     : 고정 블록 크기 (바이트)
 *  @param outPtrs   : 결과 블록 포인터 배열   [device char**]    (size = M)
 *  @param M         : 변경 블록 개수
 */

__global__
void gather_ptrs(const int32_t* __restrict__ idx,
    const char*   __restrict__ base,
    std::size_t blkSz,
    char** __restrict__ outPtrs,
    std::size_t M);  
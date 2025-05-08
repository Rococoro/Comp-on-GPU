#include "MurmurHash2.h"
#include "diff_kernel.cuh"
#include <vector>
#include <random>
#include <cassert>
#include <iostream>
#include <cuda_runtime.h>

constexpr size_t BLK = 64*1024;
constexpr int    THREADS = 256;

extern __global__
void hash_kernel(const char*, uint64_t*, size_t, size_t);   // 이미 구현됨

int main() {
    size_t N = 1024;              // 64 MiB 샘플
    size_t total = N * BLK;

    // t0, t1 데이터 준비
    std::vector<uint8_t> h_t0(total), h_t1(total);
    std::mt19937 gen(1); std::uniform_int_distribution<int> rnd(0,255);
    for(auto& v : h_t0) v = rnd(gen);
    h_t1 = h_t0;                  // 일단 동일
    // 일부 블록 수정
    for(int b : {123, 456}) h_t1[b*BLK] ^= 0xAA;

    // GPU 메모리
    char *d_t0, *d_t1; uint64_t *d_h0, *d_h1; uint8_t *d_flag;
    cudaMalloc(&d_t0, total); cudaMalloc(&d_t1, total);
    cudaMalloc(&d_h0, N*sizeof(uint64_t));
    cudaMalloc(&d_h1, N*sizeof(uint64_t));
    cudaMalloc(&d_flag, N);

    cudaMemcpy(d_t0, h_t0.data(), total, cudaMemcpyHostToDevice);
    cudaMemcpy(d_t1, h_t1.data(), total, cudaMemcpyHostToDevice);

    // 해시 계산
    dim3 block(THREADS), grid((N+THREADS-1)/THREADS);
    hash_kernel<<<grid,block>>>(d_t0,d_h0,BLK,N);
    hash_kernel<<<grid,block>>>(d_t1,d_h1,BLK,N);

    // diff kernel 실행
    launch_diff(d_h0,d_h1,d_flag,N);
    cudaDeviceSynchronize();

    // 결과 확인
    std::vector<uint8_t> h_flag(N);
    cudaMemcpy(h_flag.data(), d_flag, N, cudaMemcpyDeviceToHost);

    assert(h_flag[123]==1 && h_flag[456]==1);
    for(size_t i=0;i<N;++i)
        if(i!=123 && i!=456) assert(h_flag[i]==0);

    std::cout << "diff kernel passed\n";
    cudaFree(d_t0); cudaFree(d_t1); cudaFree(d_h0); cudaFree(d_h1); cudaFree(d_flag);
}

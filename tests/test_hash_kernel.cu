#include "MurmurHash2.h"
#include <cuda_runtime.h>
#include <cassert>
#include <random>
#include <iostream>

constexpr size_t BLK = 64 * 1024; //64KiB
constexpr int    THREADS = 256;

__global__
void hash_kernel(const char* d_in, uint64_t* d_out,
                 size_t blk_sz, size_t N);

int main() {
    size_t N = 1024;             // 64KiB * 1024 = 64 MiB 테스트
    size_t total = N * BLK;

    /* 1. host input */
    std::vector<uint8_t> h_in(total);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0,255);
    for(auto& v : h_in) v = dist(rng);

    /* 2. CPU hashes */
    std::vector<uint64_t> h_ref(N);
    for(size_t i=0;i<N;++i)
        h_ref[i] = MurmurHash64A(h_in.data() + i*BLK, BLK, 0x1e35a7bd);

    /* 3. device alloc/copy */
    char    *d_in;   cudaMalloc(&d_in, total);
    uint64_t*d_out;  cudaMalloc(&d_out, N*sizeof(uint64_t));
    cudaMemcpy(d_in, h_in.data(), total, cudaMemcpyHostToDevice);

    /* 4. kernel launch */
    dim3 block(THREADS);
    dim3 grid((N + THREADS - 1)/THREADS);
    hash_kernel<<<grid, block>>>(d_in, d_out, BLK, N);
    cudaDeviceSynchronize();

    /* 5. copy back + compare */
    std::vector<uint64_t> h_gpu(N);
    cudaMemcpy(h_gpu.data(), d_out, N*sizeof(uint64_t), cudaMemcpyDeviceToHost);

    for(size_t i=0;i<N;++i) {
        assert(h_gpu[i] == h_ref[i] && "Hash mismatch!");
    }
    std::cout << "All hashes match\n";

    /* 6. (opt) performance */
    // cudaEvent timing 코드 생략 – 필요시 추가

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
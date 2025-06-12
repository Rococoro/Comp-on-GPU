// =======================================================
// delta-compression with cuSZ (Jetson Orin, CUDA 11.4)
// =======================================================
#include <cuda_runtime.h>
#include <cusz.h>

#include "hash_kernel.cuh"
#include "gather_kernel.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <iomanip>

// -------------------------------------------------------
// 전역 설정
// -------------------------------------------------------
constexpr size_t BLK_BYTES = 64 * 1024;          // 64 KiB per block
constexpr size_t BLK       = BLK_BYTES / sizeof(float);
constexpr int    THREADS   = 256;

// -------------------------------------------------------
// 메인
// -------------------------------------------------------
int main(int argc, char** argv)
{
    // ───────── 0. cuSZ 초기화 ─────────
    auto* framework = cusz_default_framework();
    auto* comp      = cusz_create(framework, FP32);
    auto* config    = new cusz_config{
        .eb   = 1e-4f,           // ← lossy 예시 (절대오차 1e-4). 0.0이면 lossless
        .mode = CUSZ_MODE_ABS
    };

    // ───────── 1. 테스트 데이터 생성 ─────────
    const size_t N      = 1024;                // 블록 수
    const size_t total  = N * BLK;             // float 개수

    std::vector<float> h_t0(total), h_t1(total);

    std::mt19937 rng(7);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& v : h_t0) v = static_cast<float>(dist(rng));
    h_t1 = h_t0;

    // 30 % 블록 중 한 바이트만 랜덤 뒤집기
    double pct = (argc >= 2) ? std::atof(argv[1]) : 0.30;
    pct = std::clamp(pct, 0.0, 1.0);
    const size_t changeN = static_cast<size_t>(N * pct);

    std::unordered_set<int> chosen;
    std::uniform_int_distribution<int> distBlk(0, N - 1);
    std::uniform_int_distribution<int> distByte(0, BLK - 1);

    while (chosen.size() < changeN) chosen.insert(distBlk(rng));
    auto* h1_bytes = reinterpret_cast<uint8_t*>(h_t1.data());
    for (int b : chosen) {
        const size_t byte_off =
            (static_cast<size_t>(b) * BLK + distByte(rng)) * sizeof(float);
        h1_bytes[byte_off] ^= 0x5A;
    }
    std::cout << "Changed blocks : " << chosen.size() << "/" << N << '\n';

    // ───────── 2. GPU 버퍼 ─────────
    float *d_t0, *d_t1;
    uint64_t *d_h0, *d_h1;
    uint8_t* d_flag;
    cudaMalloc(&d_t0, total * sizeof(float));
    cudaMalloc(&d_t1, total * sizeof(float));
    cudaMalloc(&d_h0, N * sizeof(uint64_t));
    cudaMalloc(&d_h1, N * sizeof(uint64_t));
    cudaMalloc(&d_flag, N);

    cudaMemcpy(d_t0, h_t0.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t1, h_t1.data(), total * sizeof(float), cudaMemcpyHostToDevice);

    // ───────── 3. 해시 & diff ─────────
    dim3 block(THREADS), grid((N + THREADS - 1) / THREADS);
    hash_kernel<<<grid, block>>>(d_t0, d_h0, BLK, N);
    hash_kernel<<<grid, block>>>(d_t1, d_h1, BLK, N);
    launch_diff(d_h0, d_h1, d_flag, N);

    // ───────── 4. 변경 인덱스 추출 ─────────
    thrust::device_vector<int> deltaIdx;
    const size_t deltaCnt = dz::deltaFilter(d_flag, N, deltaIdx);

    // ───────── 5. 블록 포인터 배열 생성 ─────────
    thrust::device_vector<float*> d_ptrs(deltaCnt);
    gather_ptrs<<<(deltaCnt + THREADS - 1) / THREADS, THREADS>>>(
        deltaIdx.data().get(), d_t1, BLK, d_ptrs.data().get(), deltaCnt);

    // ───────── 6. cuSZ 압축 (블록별) ─────────
    std::vector<cusz_header> h_hdr(deltaCnt);
    std::vector<size_t>      h_outSz(deltaCnt);
    std::vector<uint8_t*>    h_compPtr(deltaCnt);

    thrust::host_vector<int>    h_delta = deltaIdx;   // D→H 복사
    thrust::host_vector<float*> h_ptrs  = d_ptrs;

    for (size_t k = 0; k < deltaCnt; ++k) {
        float*   src  = h_ptrs[k];   // device addr
        uint8_t* out  = nullptr;
        size_t   len  = 0;

        cusz_compress(
            comp, config,
            src,
            {BLK, 1, 1, 1, 1.0},      // 1-D 배열
            &out, &len,
            &h_hdr[k],
            nullptr, 0);

        h_compPtr[k] = out;
        h_outSz[k]   = len;
    }
    cudaDeviceSynchronize();

    const size_t delta_bytes = std::accumulate(
        h_outSz.begin(), h_outSz.end(), size_t{0});
    std::cout << "Compressed size (bytes): " << delta_bytes << "\n"
              << "Compression ratio: " << std::fixed << std::setprecision(2)
              << static_cast<double>(total * sizeof(float)) / delta_bytes
              << " : 1\n";

    // ───────── 7. 복원 & 검증 ─────────
    float* d_recon;
    cudaMalloc(&d_recon, total * sizeof(float));

    for (size_t k = 0; k < deltaCnt; ++k) {
        const size_t blk_idx = h_delta[k];
        float* dst = d_recon + blk_idx * BLK;

        cusz_decompress(
            comp,
            &h_hdr[k],
            h_compPtr[k], h_outSz[k],
            dst,
            {BLK, 1, 1, 1, 1.0},
            nullptr, 0);
    }
    // 변경되지 않은 블록은 그대로 복사
    for (size_t i = 0; i < N; ++i)
        if (!d_flag[i])   // host 복사 전에 flag 값 재사용하려면 별도 배열 필요
            cudaMemcpyAsync(d_recon + i * BLK, d_t0 + i * BLK,
                            BLK * sizeof(float), cudaMemcpyDeviceToDevice);

    std::vector<float> recon(total);
    cudaMemcpy(recon.data(), d_recon, total * sizeof(float),
               cudaMemcpyDeviceToHost);

    // 검증
    bool ok = std::equal(recon.begin(), recon.end(), h_t1.begin(),
                         [](float a, float b){ return fabsf(a - b) < 1e-3f; });
    std::cout << (ok ? "Round-trip OK\n" : "Mismatch\n");

    // ───────── 8. 정리 ─────────
    cudaFree(d_t0); cudaFree(d_t1);
    cudaFree(d_h0); cudaFree(d_h1);
    cudaFree(d_flag); cudaFree(d_recon);
    for (auto p : h_compPtr) cudaFree(p);

    cusz_delete(comp);
    delete config;
    return 0;
}

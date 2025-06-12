#include <cuda_runtime.h>
#include <cusz.h>

#include "hash_kernel.cuh"
#include "gather_kernel.cuh"

#include <iomanip>
#include <unordered_set>
#include <thrust/device_vector.h>
#include <vector>
#include <random>
#include <cassert>
#include <iostream>
#include <numeric>
//---------------------------------------------------------
// 전역 설정
//---------------------------------------------------------
constexpr size_t BLK_BYTES = 64 * 1024;
constexpr size_t BLK       = BLK_BYTES / sizeof(float);
constexpr int    THREADS  = 256;
const     uint64_t SEED   = 0x1e35a7bdUL;

//---------------------------------------------------------
// 파이프라인 테스트
//---------------------------------------------------------
int main(int argc, char** argv)
{
    cusz_framework* framework = cusz_default_framework();
    cusz_compressor* comp     = cusz_create(framework, FP32);
    cusz_config*     config   = new cusz_config{
        .eb   = 0.0,      // lossless: zero error bound
        .mode = CUSZ_MODE_ABS       // 절대오차 모드
    };
    //-----------------------------------------------------------------
    // 0)  시계열 t0 / t1 원본 준비  (64 MiB 예시)
    //-----------------------------------------------------------------
    size_t N = 1024;                 // 블록 수
    size_t total = N * BLK;

    std::vector<float> h_t0(total), h_t1(total);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> dist(0,255);
    for(auto& v : h_t0) v = dist(rng);
    h_t1 = h_t0;
    // 일부 블록 바꿔 보자
    double pct = 0.30;                    // 기본값 30 %
    if (argc >= 2) pct = std::atof(argv[1]);
    if (pct < 0 || pct > 1) {
        std::cerr << "pct must be 0~1\n"; return 1;
    }
    size_t changeN = static_cast<size_t>(N * pct);
    std::unordered_set<int> chosen;

    //std::mt19937 rng(42);
    std::uniform_int_distribution<int> distBlk(0, N - 1);
    std::uniform_int_distribution<int> distByte(0, BLK-1);

    while (chosen.size() < changeN) {
        chosen.insert(distBlk(rng));
    }
    for (int b : chosen) {
        size_t off = static_cast<size_t>(b) * BLK + distByte(rng);
        auto* bytes = reinterpret_cast<uint8_t*>(h_t1.data());
        bytes[off] ^= 0x5A;
    }
    std::cout << "Changed blocks : " << chosen.size() << "/" << N << '\n';

    //-----------------------------------------------------------------
    // 1)  GPU 버퍼
    //-----------------------------------------------------------------
    float *d_t0, *d_t1;
    uint64_t *d_h0, *d_h1;
    uint8_t  *d_flag;
    cudaMalloc(&d_t0, total * sizeof(float));
    cudaMalloc(&d_t1, total * sizeof(float));
    cudaMalloc(&d_h0, N*sizeof(uint64_t));
    cudaMalloc(&d_h1, N*sizeof(uint64_t));
    cudaMalloc(&d_flag, N);

    cudaMemcpy(d_t0, h_t0.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t1, h_t1.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    //-----------------------------------------------------------------
    // 2)  t0 해시 (전체)   ->  d_h0
    //-----------------------------------------------------------------
    dim3 block(THREADS), grid((N+THREADS-1)/THREADS);
    hash_kernel<<<grid,block>>>(d_t0, d_h0, BLK, N);

    //-----------------------------------------------------------------
    // 3)  t1 해시  ->  diff  ->  flag
    //-----------------------------------------------------------------
    hash_kernel<<<grid,block>>>(d_t1, d_h1, BLK, N);
    launch_diff(d_h0, d_h1, d_flag, N);                    // diff_kernel.cuh inline

    //-----------------------------------------------------------------
    // 4)  deltaFilter (Thrust copy_if) : 변경 블록 목록
    //-----------------------------------------------------------------
    thrust::device_vector<int> deltaIdx;
    size_t deltaCnt = dz::deltaFilter(d_flag, N, deltaIdx);
    
    //-----------------------------------------------------------------
    // 5)  gather_ptrs : 포인터 배열
    //-----------------------------------------------------------------
    
    thrust::device_vector<float*> d_ptrs(deltaCnt);
    gather_ptrs<<<(deltaCnt+THREADS-1)/THREADS, THREADS>>>(
        deltaIdx.data().get(),
        d_t1,                    // t1 원본 (float*)
        BLK,
        d_ptrs.data().get(),
        deltaCnt);

    thrust::device_vector<size_t> d_sizes(deltaCnt, BLK);

    //-----------------------------------------------------------------
    // 6)  nvCOMP Batched Zstd 압축 (변경 블록만)
    //-----------------------------------------------------------------
    std::vector<cusz_header>          h_hdr(deltaCnt);
    thrust::device_vector<uint8_t*>   d_compPtr_device(deltaCnt);
    std::vector<size_t>               h_outSz(deltaCnt);
    // D→H 복사
    thrust::host_vector<int>    h_delta  = deltaIdx;
    thrust::host_vector<float*> h_ptrs   = d_ptrs;
    std::vector<uint8_t*>       h_compPtr(deltaCnt); 
    for (size_t k = 0; k < deltaCnt; ++k) {
        float* src = h_ptrs[k];           // 변경된 블록(Device)
        uint8_t* out; size_t outLen;
    
        cusz_compress(comp, config,
                      src,
                      {BLK,1,1,1,1.0},    // 1-D 길이 = BLK floats
                      &out, &outLen,
                      &h_hdr[k],
                      nullptr, 0);
    
        h_compPtr[k] = out;
        d_compPtr_device[k] = out;
        h_outSz[k] = outLen;
    }
    cudaDeviceSynchronize();

    //-----------------------------------------------------------------
    // 7)  (데모용) GPU→Host 로 가져와 복원 검증
    //-----------------------------------------------------------------
    // 7‑1 헤더 플래그 host 복사
    std::vector<uint8_t> h_flag(N);
    cudaMemcpy(h_flag.data(), d_flag, N, cudaMemcpyDeviceToHost);
    
    size_t delta_bytes = std::accumulate(h_outSz.begin(), h_outSz.end(), 0ull);
    
    size_t final_bytes = delta_bytes;        // v0.1 컨테이너 헤더 무시
    double ratio = static_cast<double>(total) / final_bytes;
    
    std::cout << "Compressed size (bytes): " << final_bytes << '\n'
                << "Compression ratio: " << std::fixed << std::setprecision(2)
                << ratio << " : 1\n";
    std::vector<float> recon(total);

    // 7‑3 복원: flag==0 → t0 블록 비트스트림 사용, flag==1 → 방금 압축분 디코드
    float* d_recon;
    cudaMalloc(&d_recon, total * sizeof(float));
    
    for (size_t i = 0; i < deltaCnt; ++i) {
        size_t blkIdx   = h_delta[i];
        uint8_t* inPtr  = h_compPtr[i];
        size_t   inLen  = h_outSz[i];
        float*   dst    = d_recon + blkIdx * BLK;
    
        cusz_decompress(comp,
                        &h_hdr[i],
                        inPtr, inLen,
                        dst,
                        {BLK,1,1,1,1.0},
                        nullptr, 0);
    }
    
    // flag==0 블록은 t0 그대로 복사
    for (size_t i = 0; i < N; ++i)
        if (h_flag[i] == 0)
            cudaMemcpyAsync(d_recon + i*BLK, d_t0 + i*BLK,
                            BLK*sizeof(float), cudaMemcpyDeviceToDevice);
    
    cudaMemcpy(recon.data(), d_recon,
               total*sizeof(float), cudaMemcpyDeviceToHost);
    //-----------------------------------------------------------------
    // 8)  검증
    //-----------------------------------------------------------------
    if(recon == h_t1)
        std::cout << "Round‑trip OK\n";
    else
        std::cout << "Mismatch\n";

    // Cleanup
    cudaFree(d_t0); cudaFree(d_t1);
    cudaFree(d_h0); cudaFree(d_h1); 
    cudaFree(d_flag);
    for (auto ptr : h_compPtr) cudaFree(ptr);
    
    // cuSZ 객체 해제
    cusz_delete(comp);
    cusz_delete(framework);
    delete config;
}

#include <cuda_runtime.h>
#include "zstd.h"
#include "nvcomp/zstd.h"

#include "MurmurHash2.cuh"
#include "diff_kernel.cuh"
#include "delta_filter.cuh"
#include "gather_kernel.cuh"

#include <iomanip>
#include <unordered_set>
#include <thrust/device_vector.h>
#include <vector>
#include <random>
#include <cassert>
#include <iostream>

//---------------------------------------------------------
// 전역 설정
//---------------------------------------------------------
constexpr size_t BLK      = 64 * 1024;
constexpr int    THREADS  = 256;
const     uint64_t SEED   = 0x1e35a7bdUL;

// 단순 CPU‑Zstd 래퍼 (디코드 확인용)
std::vector<char> cpu_decompress(const std::vector<char>& comp, size_t orig)
{
    std::vector<char> out(orig);
    size_t sz = ZSTD_decompress(out.data(), orig, comp.data(), comp.size());
    assert(!ZSTD_isError(sz));
    return out;
}

//---------------------------------------------------------
// 파이프라인 테스트
//---------------------------------------------------------
int main(int argc, char** argv)
{
    //-----------------------------------------------------------------
    // 0)  시계열 t0 / t1 원본 준비  (64 MiB 예시)
    //-----------------------------------------------------------------
    size_t N = 1024;                 // 블록 수
    size_t total = N * BLK;

    std::vector<uint8_t> h_t0(total), h_t1(total);
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
        h_t1[off] ^= 0x5A;                // 한 바이트만 뒤집어도 해시 달라짐
    }
    std::cout << "Changed blocks : " << chosen.size() << "/" << N << '\n';

    //-----------------------------------------------------------------
    // 1)  GPU 버퍼
    //-----------------------------------------------------------------
    char *d_t0, *d_t1;
    uint64_t *d_h0, *d_h1;
    uint8_t  *d_flag;
    cudaMalloc(&d_t0, total);
    cudaMalloc(&d_t1, total);
    cudaMalloc(&d_h0, N*sizeof(uint64_t));
    cudaMalloc(&d_h1, N*sizeof(uint64_t));
    cudaMalloc(&d_flag, N);

    cudaMemcpy(d_t0, h_t0.data(), total, cudaMemcpyHostToDevice);
    cudaMemcpy(d_t1, h_t1.data(), total, cudaMemcpyHostToDevice);
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
    thrust::device_vector<char*> d_ptrs(deltaCnt);
    gather_ptrs<<<(deltaCnt+THREADS-1)/THREADS, THREADS>>>(
        deltaIdx.data().get(),
        d_t1,                    // t1 원본
        BLK,
        d_ptrs.data().get(),
        deltaCnt);

    thrust::device_vector<size_t> d_sizes(deltaCnt, BLK);

    //-----------------------------------------------------------------
    // 6)  nvCOMP Batched Zstd 압축 (변경 블록만)
    //-----------------------------------------------------------------
    nvcompBatchedZstdOpts_t opts = nvcompBatchedZstdDefaultOpts;

    size_t tmpBytes;
    nvcompBatchedZstdCompressGetTempSize(deltaCnt, BLK, opts, &tmpBytes);
    void* d_temp; cudaMalloc(&d_temp, tmpBytes);

    size_t maxOut;
    nvcompBatchedZstdCompressGetMaxOutputChunkSize(BLK, opts, &maxOut);

    char* d_compBuf; //전체 출력 버퍼
    cudaMalloc(&d_compBuf, deltaCnt * maxOut);
    
    std::vector<char*> h_outPtrs(deltaCnt); //host의 포인트 배열 채우기
    for (size_t k = 0; k < deltaCnt; ++k) h_outPtrs[k] = d_compBuf + k * maxOut;
    
    thrust::device_vector<char*>  d_outPtrs = h_outPtrs;   // H→D 복사
    thrust::device_vector<size_t> d_outSizes(deltaCnt);    // 결과 크기

    nvcompBatchedZstdCompressAsync(
        reinterpret_cast<const void *const *>(d_ptrs.data().get()),
        d_sizes.data().get(),
        BLK,
        deltaCnt,
        d_temp,
        tmpBytes,
        reinterpret_cast<void *const *>(d_outPtrs.data().get()),
        d_outSizes.data().get(),
        opts,
        0);                             // default stream
    cudaDeviceSynchronize();

    //-----------------------------------------------------------------
    // 7)  (데모용) GPU→Host 로 가져와 복원 검증
    //-----------------------------------------------------------------
    // 7‑1 헤더 플래그 host 복사
    std::vector<uint8_t> h_flag(N);
    cudaMemcpy(h_flag.data(), d_flag, N, cudaMemcpyDeviceToHost);

    // 7‑2 압축 결과 블록을 Host 메모리로 모으기
    std::vector<char> h_compAll;
    std::vector<size_t> h_outSz(deltaCnt);
    cudaMemcpy(h_outSz.data(), d_outSizes.data().get(),
               deltaCnt*sizeof(size_t), cudaMemcpyDeviceToHost);

    size_t delta_bytes = 0;
    for(size_t sz : h_outSz) delta_bytes += sz;
    
    size_t final_bytes = delta_bytes;        // v0.1 컨테이너 헤더 무시
    double ratio = static_cast<double>(total) / final_bytes;
    
    std::cout << "Compressed size (bytes): " << final_bytes << '\n'
                << "Compression ratio: " << std::fixed << std::setprecision(2)
                << ratio << " : 1\n";
    for(size_t k=0;k<deltaCnt;++k)
    {
        std::vector<char> tmp(h_outSz[k]);
        cudaMemcpy(tmp.data(), d_outPtrs[k], tmp.size(), cudaMemcpyDeviceToHost);
        h_compAll.insert(h_compAll.end(), tmp.begin(), tmp.end());
    }

    // 7‑3 복원: flag==0 → t0 블록 비트스트림 사용, flag==1 → 방금 압축분 디코드
    std::vector<uint8_t> recon(total);
    size_t compCursor = 0;
    for(size_t i=0;i<N;++i)
    {
        if(h_flag[i]==0)  // 재사용
        {
            // t0 블록 그대로 복사
            std::copy_n(h_t0.data()+i*BLK, BLK, recon.data()+i*BLK);
        }
        else              // 새 압축 -> 디코드
        {
            size_t sz = h_outSz.front();          // FIFO 꺼내듯
            h_outSz.erase(h_outSz.begin());
            std::vector<char> oneComp(h_compAll.begin()+compCursor,
                                      h_compAll.begin()+compCursor+sz);
            compCursor += sz;
            auto decomp = cpu_decompress(oneComp, BLK);
            std::copy_n(decomp.data(), BLK, recon.data()+i*BLK);
        }
    }
    //-----------------------------------------------------------------
    // 8)  검증
    //-----------------------------------------------------------------
    if(recon == h_t1)
        std::cout << "Round‑trip OK\n";
    else
        std::cout << "Mismatch\n";

    // Cleanup
    cudaFree(d_t0); cudaFree(d_t1);
    cudaFree(d_h0); cudaFree(d_h1); cudaFree(d_flag); cudaFree(d_temp); cudaFree(d_compBuf);
}

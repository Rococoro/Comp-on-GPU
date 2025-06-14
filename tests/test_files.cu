// tests/pf_delta_pipe.cu
#include <cuda_runtime.h>
#include "nvcomp/zstd.h"
#include "zstd.h"

#include "MurmurHash2.cuh"
#include "diff_kernel.cuh"
#include "delta_filter.cuh"
#include "gather_kernel.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

//--------------------------------------------------
// 전역 설정
//--------------------------------------------------
constexpr size_t BLK     = 64 * 1024; // 64 KiB
constexpr int    THREADS = 256;

//--------------------------------------------------
// 유틸: 파일 전체 읽기
//--------------------------------------------------
std::vector<uint8_t> read_whole(const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("cannot open "+path);

    ifs.seekg(0, std::ios::end);
    size_t sz = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);

    std::vector<uint8_t> buf(sz);
    ifs.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

//--------------------------------------------------
// main
//--------------------------------------------------
int main()
{
    //------------------------------------------------------------------
    // 1) 스냅숏 목록 Pf01.bin .. Pf09.bin
    //------------------------------------------------------------------
    std::vector<std::string> files;
    for (int i = 1; i <= 9; ++i) {
        files.emplace_back("./dataset/Pf0" + std::to_string(i) + ".bin");
    }

    //------------------------------------------------------------------
    // 2) 공용 CUDA 리소스
    //------------------------------------------------------------------
    cudaStream_t stream;  cudaStreamCreate(&stream);

    thrust::device_vector<uint64_t> d_hash, d_hash_prev;
    thrust::device_vector<uint8_t>  d_flag;
    thrust::device_vector<int32_t>  d_deltaIdx;

    thrust::device_vector<char*>  d_inPtrs;   // 압축 대상 블록 포인터
    thrust::device_vector<char*>  d_outPtrs;  // 출력 포인터
    thrust::device_vector<size_t> d_outBytes; // 출력 길이

    // nvCOMP 임시 버퍼
    nvcompBatchedZstdOpts_t opts = nvcompBatchedZstdDefaultOpts;
    size_t compTempBytes = 0;   // 처음 호출 시 크기 파악
    void*  d_compTemp    = nullptr;

    bool   first = true;
    size_t compBufMax = 0;      // d_compBuf 현재 용량
    char*  d_compBuf  = nullptr;

    //------------------------------------------------------------------
    // 3) 스냅숏 순회
    //------------------------------------------------------------------
    for (size_t step = 0; step < files.size(); ++step)
    {
        // 3-1. 파일 로드 → GPU 복사
        auto   h_buf = read_whole(files[step]);
        size_t bytes = h_buf.size();
        size_t padded_bytes = ((bytes + BLK - 1) / BLK) * BLK;
        std::vector<uint8_t> h_buf_padded(padded_bytes, 0);      // 남는 부분 0으로 채움
        std::memcpy(h_buf_padded.data(), h_buf.data(), bytes);   // 원본 복사
        const uint8_t* h_raw = h_buf_padded.data();
        size_t N = padded_bytes / BLK;                           // 이제 항상 정수 배

        char* d_raw;
        cudaMalloc(&d_raw, padded_bytes);
        cudaMemcpyAsync(d_raw, h_raw, padded_bytes,
                        cudaMemcpyHostToDevice, stream);

        // 해시/flag 버퍼 크기 맞추기
        d_hash.resize(N);
        d_flag.resize(N);

        // 3-2. 해시 계산
        dim3 block(THREADS);
        dim3 grid((N + THREADS - 1) / THREADS);
        hash_kernel<<<grid, block, 0, stream>>>(
            d_raw,
            thrust::raw_pointer_cast(d_hash.data()),
            BLK, N);

        //------------------------------------------------------------------
        // 4) 첫 스냅숏이면 전체 압축
        //------------------------------------------------------------------
        if (first)
        {
            // 입력 포인터 배열
            std::vector<char*> h_inPtr(N);
            for (size_t i = 0; i < N; ++i) h_inPtr[i] = d_raw + i * BLK;
            d_inPtrs  = h_inPtr;

            // 출력 버퍼 준비
            size_t maxOut; nvcompBatchedZstdCompressGetMaxOutputChunkSize(
                               BLK, opts, &maxOut);
            if (compBufMax < N*maxOut) {
                if (d_compBuf) cudaFree(d_compBuf);
                cudaMalloc(&d_compBuf, N * maxOut);
                compBufMax = N * maxOut;
            }
            std::vector<char*> h_outPtr(N);
            for (size_t i = 0; i < N; ++i) h_outPtr[i] = d_compBuf + i * maxOut;
            d_outPtrs  = h_outPtr;
            d_outBytes.resize(N);

            // 임시 버퍼 (한 번만)
            if (compTempBytes == 0) {
                nvcompBatchedZstdCompressGetTempSize(N, BLK, opts, &compTempBytes);
                cudaMalloc(&d_compTemp, compTempBytes);
            }

            nvcompBatchedZstdCompressAsync(
                reinterpret_cast<const void* const*>(d_inPtrs .data().get()),
                /*d_in_bytes=*/nullptr,
                BLK,
                N,
                d_compTemp, compTempBytes,
                reinterpret_cast<void* const*>(d_outPtrs.data().get()),
                d_outBytes.data().get(),
                opts,
                stream);
            cudaStreamSynchronize(stream);

            std::cout << "[t0] Compressed "
                      << std::fixed << std::setprecision(2)
                      << bytes/1.0e6 << " MB\n";
            first = false;
        }
        //------------------------------------------------------------------
        // 5) 이후 스냅숏: Δ 파이프라인
        //------------------------------------------------------------------
        else
        {
            // 5-1 diff → flag
            diff_kernel<<<grid, block, 0, stream>>>(
                thrust::raw_pointer_cast(d_hash_prev.data()),
                thrust::raw_pointer_cast(d_hash.data()),
                thrust::raw_pointer_cast(d_flag.data()),
                N);

            // 5-2 flag==1 인덱스만 추출
            d_deltaIdx.resize(N);
            size_t deltaCnt = dz::deltaFilter(
                                thrust::raw_pointer_cast(d_flag.data()),
                                N, d_deltaIdx, stream);

            std::cout << "Changed blocks : "
                      << deltaCnt << "/" << N << '\n';

            if (deltaCnt == 0) {
                d_hash_prev = d_hash;
                cudaFree(d_raw);
                continue; // 변화 없음
            }

            // 5-3 변경 블록 포인터 모으기
            d_inPtrs.resize(deltaCnt);
            size_t shmem = THREADS * sizeof(char*);
            gather_ptrs<<< (deltaCnt+THREADS-1)/THREADS, THREADS,
                           shmem, stream>>>(
                d_deltaIdx.data().get(),
                d_raw, BLK,
                d_inPtrs.data().get(),
                deltaCnt);

            // 5-4 출력 버퍼 준비
            size_t maxOut; nvcompBatchedZstdCompressGetMaxOutputChunkSize(
                               BLK, opts, &maxOut);
            if (compBufMax < deltaCnt*maxOut) {
                if (d_compBuf) cudaFree(d_compBuf);
                cudaMalloc(&d_compBuf, deltaCnt * maxOut);
                std::vector<char*> h_out(deltaCnt);
                for(size_t i=0;i<deltaCnt;++i) h_out[i] = d_compBuf + i*maxOut;
                thrust::device_vector<char*> d_outPtrs(h_out.begin(), h_out.end());
            }
            std::vector<char*> h_out(deltaCnt);
            for (size_t i = 0; i < deltaCnt; ++i)
                h_out[i] = d_compBuf + i * maxOut;
            d_outPtrs  = h_out;
            d_outBytes.resize(deltaCnt);

            // 5-5 nvCOMP 압축
            nvcompBatchedZstdCompressAsync(
                reinterpret_cast<const void* const*>(d_inPtrs.data().get()),
                /*d_in_bytes=*/nullptr,
                BLK,
                deltaCnt,
                d_compTemp, compTempBytes,
                reinterpret_cast<void* const*>(d_outPtrs.data().get()),
                d_outBytes.data().get(),
                opts,
                stream);
            cudaStreamSynchronize(stream);

            // 압축 바이트 합계
            thrust::host_vector<size_t> h_sz(d_outBytes);
            size_t compBytes = std::accumulate(h_sz.begin(), h_sz.end(), 0ULL);

            std::cout << "Compressed size (bytes): " << compBytes << '\n'
                      << "Compression ratio      : "
                      << std::fixed << std::setprecision(2)
                      << double(bytes) / compBytes << " : 1\n";
        }

        //------------------------------------------------------------------
        // 6) 다음 스텝 준비
        //------------------------------------------------------------------
        d_hash_prev = d_hash;
        cudaFree(d_raw);
    }

    //------------------------------------------------------------------
    // 종료
    //------------------------------------------------------------------
    cudaStreamDestroy(stream);
    if (d_compBuf) cudaFree(d_compBuf);
    if (d_compTemp) cudaFree(d_compTemp);

    return 0;
}

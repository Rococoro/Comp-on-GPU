#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <nvcomp/zstd.h>
#include "hash_kernel.cuh"
#include "gather_kernel.cuh"

constexpr int    THREADS = 256;
constexpr size_t BLK     = 64 * 1024;          // 64 KiB

/*----------- 유틸 : 파일 전체 읽기 ----------------*/
std::vector<uint8_t> read_file(const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("cannot open "+path);
    ifs.seekg(0, std::ios::end);
    size_t sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(sz);
    ifs.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

/*----------- main ----------------*/
int main()
{
    /* 1. 스냅숏 파일 목록 */
    std::vector<std::string> files;
    for(int i=0;i<=4;++i){
        files.emplace_back("/dataset/snapshot_0"+std::to_string(i)+".h5");
    }

    /* 2. GPU 리소스 공통 준비 */
    cudaStream_t stream;  cudaStreamCreate(&stream);
    thrust::device_vector<uint64_t> d_hash, d_hash_prev;
    thrust::device_vector<uint8_t>  d_flag;
    thrust::device_vector<int32_t>  d_deltaIdx;
    thrust::device_vector<char*>    d_ptrs;
    thrust::device_vector<size_t>   d_sizes;

    std::vector<char*>   h_outPtr;
    std::vector<size_t>  h_outSz;

    size_t compTempBytes = 0;
    void*  d_compTemp    = nullptr;

    bool first = true;
    size_t totalBytes = 0;

    for(size_t step=0; step<files.size(); ++step)
    {
        /* 3. 파일 로드 → GPU 복사 */
        auto h_buf = read_file(files[step]);
        size_t bytes = h_buf.size();
        totalBytes   = bytes;
        size_t N     = (bytes + BLK - 1) / BLK;

        char* d_raw;
        cudaMalloc(&d_raw, bytes);
        cudaMemcpyAsync(d_raw, h_buf.data(), bytes, cudaMemcpyHostToDevice, stream);

        /* 3-1. 해시 버퍼 resize */
        d_hash.resize(N);  d_flag.resize(N);

        /* 4. hash_kernel */
        dim3 block(THREADS);
        dim3 grid((N + THREADS - 1)/THREADS);
        hash_kernel<<<grid, block, 0, stream>>>(d_raw, thrust::raw_pointer_cast(d_hash.data()),
                                                BLK, N);

        if(first) {
            /* 4-A. t0 → 전체 압축 */
            /* nvCOMP 준비 */
            nvcompBatchedZstdOpts_t opts = nvcompBatchedZstdDefaultOpts;
            if (compTempBytes==0)
                nvcompBatchedZstdCompressGetTempSize(N, BLK, opts, &compTempBytes),
                cudaMalloc(&d_compTemp, compTempBytes);

            /* outPtrs 초기화 */
            size_t maxOut;  nvcompBatchedZstdCompressGetMaxOutputChunkSize(BLK, opts, &maxOut);
            h_outPtr.resize(N);
            char* d_compBuf;  cudaMalloc(&d_compBuf, N*maxOut);
            for(size_t i=0;i<N;i++) h_outPtr[i] = d_compBuf + i*maxOut;

            d_ptrs   = thrust::device_vector<char*>(h_outPtr.begin(), h_outPtr.end());
            d_sizes  = thrust::device_vector<size_t>(N);

            nvcompBatchedZstdCompressAsync(
                reinterpret_cast<const void* const*>(d_ptrs.data().get()),
                nullptr, BLK, N,
                d_compTemp, compTempBytes,
                reinterpret_cast<void* const*>(d_ptrs.data().get()),
                d_sizes.data().get(), opts, stream);
            cudaStreamSynchronize(stream);

            std::cout << "\n[t0] Compressed " << bytes/1e6 << " MB\n";
            first = false;
        }
        else {
            /* 4-B. Δ-파이프라인 */

            /* diff_kernel */
            diff_kernel<<<grid, block, 0, stream>>>(thrust::raw_pointer_cast(d_hash_prev.data()),
                                                    thrust::raw_pointer_cast(d_hash.data()),
                                                    thrust::raw_pointer_cast(d_flag.data()), N);

            /* deltaFilter: 1 → index copy */
            d_deltaIdx.resize(N);
            auto endIt = thrust::copy_if(thrust::device,
                                         thrust::make_counting_iterator<int32_t>(0),
                                         thrust::make_counting_iterator<int32_t>(N),
                                         d_flag.begin(),
                                         d_deltaIdx.begin(),
                                         thrust::identity<int>());
            size_t deltaCnt = endIt - d_deltaIdx.begin();

            std::cout << "Changed blocks : " << deltaCnt << "/" << N << '\n';

            /* gather_ptrs */
            d_ptrs .resize(deltaCnt);
            d_sizes.resize(deltaCnt);
            size_t shmem = THREADS * sizeof(char*);
            gather_ptrs<<<(deltaCnt+THREADS-1)/THREADS, THREADS,
                          shmem, stream>>>(
                 d_deltaIdx.data().get(),
                 d_raw, BLK, d_ptrs.data().get(), deltaCnt);

            /* nvCOMP 압축 변경 블록만 */
            size_t maxOut; nvcompBatchedZstdCompressGetMaxOutputChunkSize(BLK, nvcompBatchedZstdDefaultOpts, &maxOut);
            char* d_compBuf; cudaMalloc(&d_compBuf, deltaCnt*maxOut);
            for(size_t i=0;i<deltaCnt;i++) h_outPtr[i] = d_compBuf + i*maxOut;
            cudaMemcpyAsync(d_ptrs.data().get(), h_outPtr.data(),
                            deltaCnt*sizeof(char*), cudaMemcpyHostToDevice, stream);

            nvcompBatchedZstdCompressAsync(
                reinterpret_cast<const void* const*>(d_ptrs.data().get()),
                nullptr, BLK, deltaCnt,
                d_compTemp, compTempBytes,
                reinterpret_cast<void* const*>(d_ptrs.data().get()),
                d_sizes.data().get(), nvcompBatchedZstdDefaultOpts, stream);

            cudaStreamSynchronize(stream);

            /* 압축 크기 합계 */
            thrust::host_vector<size_t> h_sz(d_sizes);
            size_t compBytes = 0;
            for(auto s: h_sz) compBytes += s;
            std::cout << "Compressed size (bytes): " << compBytes << '\n'
                      << "Compression ratio        : "
                      << std::fixed << std::setprecision(2)
                      << double(totalBytes) / compBytes << " : 1\n";
        }

        /* 5. swap 해시 */
        d_hash_prev = d_hash;
        cudaFree(d_raw);
    }

    cudaStreamDestroy(stream);
    cudaFree(d_compTemp);
    return 0;
}

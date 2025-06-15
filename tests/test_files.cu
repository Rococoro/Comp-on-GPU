// tests/pf_delta_pipe.cu  ----------------------------------------------
// Δ-compression pipeline (64 KiB blocks) --extended-lambda friendly
// CUDA 12.9  | nvCOMP 4  | C++17
// ----------------------------------------------------------------------

#include <cuda_runtime.h>
#include <nvcomp/zstd.h>
#include "zstd.h"

#include "MurmurHash2.cuh"   // 64-bit block hash
#include "gather_kernel.cuh"
#include "delta_filter.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <numeric>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

// ------------------------------------------------------------------
// basic helpers (inline fn → 따옴표 오류 ×)
// ------------------------------------------------------------------
inline void cuCheck(cudaError_t e,
                    const char* f, int l){
  if(e!=cudaSuccess){
    std::cerr<<"CUDA "<<cudaGetErrorString(e)
             <<" @ "<<f<<':'<<l<<'\n';
    std::exit(1);
  }
}
#define CUCHK(x) cuCheck((x),__FILE__,__LINE__)

inline void nvCheck(nvcompStatus_t s,
                    const char* f,int l){
  if(s!=nvcompSuccess){
    std::cerr<<"nvCOMP error "<<int(s)
             <<" @ "<<f<<':'<<l<<'\n';
    std::exit(1);
  }
}
#define NVCHK(x) nvCheck((x),__FILE__,__LINE__)

// ------------------------------------------------------------------
constexpr size_t BLK = 64*1024; // 64 KiB
constexpr int    THREADS = 256;     // threads/block

// ------------------------------------------------------------------
std::vector<uint8_t> read_whole(const std::string& p)
{
  std::ifstream fs(p,std::ios::binary);
  if(!fs) throw std::runtime_error("open "+p);
  fs.seekg(0,std::ios::end);
  size_t n=static_cast<size_t>(fs.tellg());
  fs.seekg(0,std::ios::beg);
  std::vector<uint8_t> buf(n);
  fs.read(reinterpret_cast<char*>(buf.data()),n);
  return buf;
}

// ==================================================================
int main()
{
    
    //-----------------------------------------------------------------
    std::vector<std::string> files;
    for(int i=1;i<=9;++i)
        files.emplace_back("./dataset/Pf0"+std::to_string(i)+".bin");

    //-----------------------------------------------------------------
    cudaStream_t stream;  CUCHK(cudaStreamCreate(&stream));
    nvcompBatchedZstdOpts_t opts = nvcompBatchedZstdDefaultOpts;

    thrust::device_vector<uint64_t> d_hash, d_prev, d_prev_sorted;
    thrust::device_vector<uint8_t>  d_flag;
    thrust::device_vector<int32_t>  d_deltaIdx;

    thrust::device_vector<char*>  d_inPtr, d_outPtr;
    thrust::device_vector<size_t> d_inSz,  d_outSz;

    void* d_temp=nullptr;  size_t tempBytes=0;
    char* d_compBuf=nullptr; size_t compCap=0;

    bool   first=true;

    //-----------------------------------------------------------------
    for(const auto& path: files)
    {
        //---------------- load & pad ----------------
        auto h_raw = read_whole(path);
        const size_t bytes  = h_raw.size();
        const size_t padded = ((bytes+BLK-1)/BLK)*BLK;
        h_raw.resize(padded,0);
        const size_t N = padded/BLK;

        char* d_raw=nullptr;
        CUCHK(cudaMalloc(&d_raw,padded));
        CUCHK(cudaMemcpyAsync(d_raw,h_raw.data(),padded,
                            cudaMemcpyHostToDevice,stream));

        //---------------- hash ----------------------
        d_hash.resize(N);
        dim3 grid((N+THREADS-1)/THREADS), block(THREADS);
        hash_kernel<<<grid,block,0,stream>>>(
            d_raw,
            thrust::raw_pointer_cast(d_hash.data()),
            BLK,N);
        CUCHK(cudaPeekAtLastError());
        CUCHK(cudaStreamSynchronize(stream));

        //---------------- 첫 스냅숏 ------------------
        if(first){
        std::cout<<"[t0] blocks="<<N<<" ("<<bytes/1e6<<" MB)\n";

        // input arrays
        std::vector<char*> h_ptr(N);
        for(size_t i=0;i<N;++i) h_ptr[i]=d_raw+i*BLK;
        d_inPtr = h_ptr;
        d_inSz.assign(N,BLK);

        // output buffer
        size_t maxOut;
        NVCHK(nvcompBatchedZstdCompressGetMaxOutputChunkSize(BLK,opts,&maxOut));
        compCap = N*maxOut; CUCHK(cudaMalloc(&d_compBuf,compCap));
        std::vector<char*> h_out(N);
        for(size_t i=0;i<N;++i) h_out[i]=d_compBuf+i*maxOut;
        d_outPtr=h_out; d_outSz.assign(N,0);

        NVCHK(nvcompBatchedZstdCompressGetTempSize(N,BLK,opts,&tempBytes));
        CUCHK(cudaMalloc(&d_temp,tempBytes));

        NVCHK(nvcompBatchedZstdCompressAsync(
            reinterpret_cast<const void* const*>(d_inPtr.data().get()),
            d_inSz.data().get(),
            BLK,N,
            d_temp,tempBytes,
            reinterpret_cast<void* const*>(d_outPtr.data().get()),
            d_outSz.data().get(),
            opts,stream));
        CUCHK(cudaStreamSynchronize(stream));

        size_t compBytes = thrust::reduce(d_outSz.begin(),d_outSz.end(),0ULL);
        std::cout<<"     compressed bytes="<<compBytes<<"\n";

        d_prev = d_hash;
        d_prev_sorted = d_prev;
        thrust::sort(d_prev_sorted.begin(),d_prev_sorted.end());
        first=false; CUCHK(cudaFree(d_raw)); continue;
        }

        //---------------- Δ-경로 ---------------------
        // flag : 1(새로운 블록) / 0(존재)
        d_flag.resize(N);
        thrust::binary_search(thrust::device,
        d_prev_sorted.begin(), d_prev_sorted.end(),
        d_hash.begin(),        d_hash.end(),
        d_flag.begin());                 // 결과 0/1 (존재? →1)

        thrust::transform(thrust::device,
        d_flag.begin(), d_flag.end(), d_flag.begin(),
        [] __device__ (uint8_t found){ return found?0u:1u; });

        // delta indices
        d_deltaIdx.resize(N);
        size_t deltaCnt = dz::deltaFilter(
            thrust::raw_pointer_cast(d_flag.data()), N,
            d_deltaIdx, stream);
        CUCHK(cudaStreamSynchronize(stream));

        std::cout<<"deltaCnt="<<deltaCnt<<" / "<<N<<'\n';
        if(deltaCnt==0){
        d_prev=d_hash;
        d_prev_sorted=d_hash;
        thrust::sort(d_prev_sorted.begin(),d_prev_sorted.end());
        CUCHK(cudaFree(d_raw)); continue;
        }

        // gather
        d_inPtr.resize(deltaCnt);
        gather_ptrs<<< (deltaCnt+THREADS-1)/THREADS, THREADS,
                    THREADS*sizeof(char*), stream>>>(
            d_deltaIdx.data().get(),
            d_raw, BLK,
            d_inPtr.data().get(),
            deltaCnt);
        CUCHK(cudaPeekAtLastError());
        CUCHK(cudaStreamSynchronize(stream));

        d_inSz.assign(deltaCnt,BLK);

        // output cap
        size_t maxOut;
        NVCHK(nvcompBatchedZstdCompressGetMaxOutputChunkSize(BLK,opts,&maxOut));
        if(compCap < deltaCnt*maxOut){
        if(d_compBuf) CUCHK(cudaFree(d_compBuf));
        compCap = deltaCnt*maxOut; CUCHK(cudaMalloc(&d_compBuf,compCap));
        }
        std::vector<char*> h_out(deltaCnt);
        for(size_t i=0;i<deltaCnt;++i) h_out[i]=d_compBuf+i*maxOut;
        d_outPtr=h_out; d_outSz.assign(deltaCnt,0);

        size_t needTemp;
        NVCHK(nvcompBatchedZstdCompressGetTempSize(deltaCnt,BLK,opts,&needTemp));
        if(needTemp>tempBytes){
        if(d_temp) CUCHK(cudaFree(d_temp));
        tempBytes=needTemp; CUCHK(cudaMalloc(&d_temp,tempBytes));
        }

        NVCHK(nvcompBatchedZstdCompressAsync(
            reinterpret_cast<const void* const*>(d_inPtr.data().get()),
            d_inSz.data().get(), BLK, deltaCnt,
            d_temp, tempBytes,
            reinterpret_cast<void* const*>(d_outPtr.data().get()),
            d_outSz.data().get(),
            opts, stream));
        CUCHK(cudaStreamSynchronize(stream));

        size_t compBytes = thrust::reduce(d_outSz.begin(),d_outSz.end(),0ULL);
        std::cout<<"   compressed bytes="<<compBytes
                <<" (ratio "<<std::fixed<<std::setprecision(2)
                <<double(bytes)/compBytes<<" : 1)\n";

        // next round
        d_prev=d_hash;
        d_prev_sorted=d_hash;
        thrust::sort(d_prev_sorted.begin(),d_prev_sorted.end());

        CUCHK(cudaFree(d_raw));
    }

    //-----------------------------------------------------------------
    CUCHK(cudaStreamDestroy(stream));
    if(d_compBuf) CUCHK(cudaFree(d_compBuf));
    if(d_temp)    CUCHK(cudaFree(d_temp));
    std::cout<<"[Done]\n";
    return 0;
}

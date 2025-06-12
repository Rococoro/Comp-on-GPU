// delta_cusz_pipeline.cpp
// Jetson‑Orin AGX | CUDA 11.4 | cuSZ 2.x
// --------------------------------------------------------------
// Δ‑compression pipeline for a sequence of HDF5 snapshots.
// Each snapshot contains a single dataset "/field" (float32, 3‑D).
// --------------------------------------------------------------
#include <cuda_runtime.h>
#include <cusz.h>

#include "hash_kernel.cuh"   // user‑supplied block‑hash kernel
#include "gather_kernel.cuh" // user‑supplied gather kernel

#include <H5Cpp.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <numeric>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

// --------------------------------------------------------------
// constants
// --------------------------------------------------------------
constexpr int    THREADS   = 256;
constexpr size_t BLK_F32   = 64 * 1024 / sizeof(float);   // 16,384 floats (64 KiB)

// --------------------------------------------------------------
// load HDF5 snapshot → std::vector<float>
// --------------------------------------------------------------
std::vector<float> load_snapshot(const std::string& path)
{
    H5::H5File file(path, H5F_ACC_RDONLY);
    H5::DataSet dset = file.openDataSet("/field");

    H5::DataSpace space = dset.getSpace();
    hsize_t dims[3];
    space.getSimpleExtentDims(dims);          // e.g. 300 × 300 × 300
    size_t nfloat = dims[0] * dims[1] * dims[2];

    std::vector<float> buf(nfloat);
    dset.read(buf.data(), H5::PredType::NATIVE_FLOAT);
    return buf;
}

// --------------------------------------------------------------
int main()
{
    // 1. file list
    std::vector<std::string> files;
    for (int i = 0; i <= 4; ++i)
        files.emplace_back("/dataset/snapshot_0" + std::to_string(i) + ".h5");

    // 2. cuSZ initialisation (Ampere, FP32)
    auto* fw   = cusz_default_framework();
    auto* comp = cusz_create(fw, FP32);
    auto* cfg  = new cusz_config{ .eb = 1e-4f, .mode = CUSZ_MODE_ABS };

    // 3. common CUDA resources
    cudaStream_t stream;  cudaStreamCreate(&stream);

    thrust::device_vector<uint64_t> d_hash, d_hash_prev;
    thrust::device_vector<uint8_t>  d_flag;
    thrust::device_vector<int32_t>  d_deltaIdx;
    thrust::device_vector<float*>   d_ptrs;

    bool first = true;

    for (const auto& snap : files)
    {
        // 3‑A. load snapshot (host)
        auto  h_buf = load_snapshot(snap);
        size_t F    = h_buf.size();              // number of floats
        size_t Nblk = (F + BLK_F32 - 1) / BLK_F32;

        // 3‑B. upload to GPU
        float* d_raw;
        cudaMallocAsync(&d_raw, F * sizeof(float), stream);
        cudaMemcpyAsync(d_raw, h_buf.data(), F * sizeof(float),
                        cudaMemcpyHostToDevice, stream);

        // 3‑C. resize hash / flag buffers
        d_hash.resize(Nblk);
        d_flag.resize(Nblk);

        dim3 block(THREADS), grid((Nblk + THREADS - 1) / THREADS);
        hash_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const char*>(d_raw),
            thrust::raw_pointer_cast(d_hash.data()),
            BLK_F32, Nblk);

        // --------------------------------------------------
        if (first) {
            // 4‑A. compress ALL blocks with cuSZ
            std::vector<cusz_header> hdr(Nblk);
            std::vector<size_t>      outsz(Nblk);
            std::vector<uint8_t*>    outptr(Nblk);

            for (size_t k = 0; k < Nblk; ++k) {
                float*   src = d_raw + k * BLK_F32;
                uint8_t* out = nullptr; size_t len = 0;
                cusz_compress(comp, cfg,
                              src,
                              {BLK_F32, 1, 1, 1, 1.0},
                              &out, &len, &hdr[k], nullptr, stream);
                outptr[k] = out;
                outsz [k] = len;
            }
            cudaStreamSynchronize(stream);

            size_t tot = std::accumulate(outsz.begin(), outsz.end(), size_t{0});
            std::cout << "[t0] " << snap << "  compressed "
                      << std::fixed << std::setprecision(1)
                      << double(tot) / (1<<20) << " MiB\n";
            first = false;

            // compressed buffers kept in GPU memory for demo; free when done
            for (auto p : outptr) cudaFreeAsync(p, stream);
        }
        else {
            // 4‑B. hash diff → flag → delta indices
            diff_kernel<<<grid, block, 0, stream>>>(
                thrust::raw_pointer_cast(d_hash_prev.data()),
                thrust::raw_pointer_cast(d_hash.data()),
                thrust::raw_pointer_cast(d_flag.data()),
                Nblk);

            d_deltaIdx.resize(Nblk);
            auto endIt = thrust::copy_if(
                thrust::device,
                thrust::make_counting_iterator<int32_t>(0),
                thrust::make_counting_iterator<int32_t>(Nblk),
                d_flag.begin(),
                d_deltaIdx.begin(),
                thrust::identity<int>());
            size_t deltaCnt = endIt - d_deltaIdx.begin();
            std::cout << "changed blocks : " << deltaCnt << "/" << Nblk << "\n";

            // gather pointers
            d_ptrs.resize(deltaCnt);
            gather_ptrs<<<(deltaCnt + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
                d_deltaIdx.data().get(),
                d_raw, BLK_F32,
                d_ptrs.data().get(), deltaCnt);

            // copy to host vectors
            thrust::host_vector<int32_t> h_idx  = d_deltaIdx;
            thrust::host_vector<float*>  h_ptrs = d_ptrs;

            std::vector<cusz_header> hdr(deltaCnt);
            std::vector<size_t>      outsz(deltaCnt);
            std::vector<uint8_t*>    outptr(deltaCnt);

            size_t total_delta_bytes = 0;
            for (size_t k = 0; k < deltaCnt; ++k) {
                float*   src = h_ptrs[k];
                uint8_t* out = nullptr; size_t len = 0;

                cusz_compress(comp, cfg, src,
                              {BLK_F32, 1, 1, 1, 1.0},
                              &out, &len, &hdr[k], nullptr, stream);
                outptr[k] = out;  outsz[k] = len;
                total_delta_bytes += len;
            }
            cudaStreamSynchronize(stream);

            double ratio = double(F * sizeof(float)) / total_delta_bytes;
            std::cout << "compressed Δ size (bytes): " << total_delta_bytes << "\n"
                      << "compression ratio         : "
                      << std::fixed << std::setprecision(2)
                      << ratio << " : 1\n";

            for (auto p : outptr) cudaFreeAsync(p, stream);
        }

        // swap hash for next step
        d_hash_prev = d_hash;
        cudaFreeAsync(d_raw, stream);
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cusz_delete(comp);
    delete cfg;
    return 0;
}

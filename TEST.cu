/**
 * zfp CUDA → CPU round-trip test
 *   – Jetson Orin/Xavier, zfp 1.0.1, CUDA 11.x 이상
 */
#include "zfp.h"
#include <cuda_runtime.h>

#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                     \
do {                                                                         \
  cudaError_t err = call;                                                    \
  if (err != cudaSuccess) {                                                  \
    fprintf(stderr,"CUDA ERROR %s:%d - %s\n",                                \
            __FILE__,__LINE__, cudaGetErrorString(err)); exit(1); }          \
} while(0)

int main()
{
  /* ────────── 0. 실험 파라미터 ────────── */
  constexpr size_t nx = 128, ny = 128, nz = 128;
  const size_t elems = nx * ny * nz;
  const size_t bytes = elems * sizeof(float);
  const double tol   = 1e-3;                 // absolute accuracy

  /* ────────── 1. 입력 데이터 생성 (Host) ────────── */
  std::vector<float> h_src(elems);
  for (size_t i = 0; i < elems; ++i)
    h_src[i] = std::sin(float(i) * 0.001f);

  /* ────────── 2. GPU 메모리로 복사 ────────── */
  float* d_src;
  CUDA_CHECK(cudaMalloc(&d_src, bytes));
  CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), bytes, cudaMemcpyHostToDevice));

  /* ────────── 3. zfp field & stream (GPU 실행) ────────── */
  zfp_field*  field_gpu = zfp_field_3d(d_src, zfp_type_float, nx, ny, nz);
  zfp_stream* zfp_gpu   = zfp_stream_open(nullptr);
  zfp_stream_set_execution(zfp_gpu, zfp_exec_cuda);
  zfp_stream_set_accuracy(zfp_gpu, tol);

  /* 3-1. 최대 압축 크기 계산 → GPU 버퍼 확보 */
  size_t maxBytes = zfp_stream_maximum_size(zfp_gpu, field_gpu);
  void*  d_comp   = nullptr;
  CUDA_CHECK(cudaMalloc(&d_comp, maxBytes));

  bitstream* bs_gpu = stream_open(d_comp, maxBytes);
  zfp_stream_set_bit_stream(zfp_gpu, bs_gpu);
  zfp_stream_rewind(zfp_gpu);

  /* ────────── 4. 압축 (GPU) ────────── */
  size_t compBytes = zfp_compress(zfp_gpu, field_gpu);
  CUDA_CHECK(cudaDeviceSynchronize());

  if (compBytes == 0) {
    printf("Compression failed (buffer too small?)\n");
    return 1;
  }
  printf("Compressed %.2f MB → %.2f MB  (%.2f : 1)\n",
         bytes/1e6, compBytes/1e6, double(bytes)/compBytes);

  /* ────────── 5. 비트스트림 Host 복사 ────────── */
  std::vector<unsigned char> h_stream(compBytes);
  CUDA_CHECK(cudaMemcpy(h_stream.data(), d_comp, compBytes,
                        cudaMemcpyDeviceToHost));

  /* ────────── 6. CPU 복원 셋업 ────────── */
  std::vector<float> h_out(elems);

  zfp_field*  field_cpu = zfp_field_3d(h_out.data(), zfp_type_float, nx, ny, nz);
  zfp_stream* zfp_cpu   = zfp_stream_open(nullptr);
  zfp_stream_set_execution(zfp_cpu, zfp_exec_serial);
  zfp_stream_set_accuracy(zfp_cpu, tol);

  bitstream* bs_cpu = stream_open(h_stream.data(), compBytes);
  zfp_stream_set_bit_stream(zfp_cpu, bs_cpu);
  zfp_stream_rewind(zfp_cpu);

  /* ────────── 7. 복원 (CPU) ────────── */
  if (!zfp_decompress(zfp_cpu, field_cpu)) {
    printf("Decompression failed\n");
    return 1;
  }

  /* ────────── 8. 최대 오차 계산 ────────── */
  double max_err = 0.0;
  for (size_t i = 0; i < elems; ++i)
    max_err = std::max(max_err, std::fabs(double(h_src[i]) - h_out[i]));
  printf("max |error| = %.3e  (tolerance %.3e)\n", max_err, tol);

  /* ────────── 9. 정리 ────────── */
  stream_close(bs_gpu);  stream_close(bs_cpu);
  zfp_stream_close(zfp_gpu); zfp_stream_close(zfp_cpu);
  zfp_field_free(field_gpu); zfp_field_free(field_cpu);
  CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_comp));
  return 0;
}

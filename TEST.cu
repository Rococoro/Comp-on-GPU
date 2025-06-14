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
    fprintf(stderr,"CUDA %s:%d  %s\n",                                       \
            __FILE__, __LINE__, cudaGetErrorString(err)); exit(1);}          \
} while(0)

int main()
{
  /* ── 파라미터 ───────────────────────────────────── */
  const size_t nx = 128, ny = 128, nz = 64;
  const size_t n  = nx * ny * nz;
  const size_t bytes = n * sizeof(float);
  const double tol   = 1e-3;                 // absolute error bound

  /* ── 원본 데이터 (Host) ──────────────────────────── */
  std::vector<float> h_src(n);
  for (size_t i = 0; i < n; ++i)
    h_src[i] = std::sin(float(i) * 0.01f);

  /* ── GPU 메모리 업로드 ───────────────────────────── */
  float* d_src;
  CUDA_CHECK(cudaMalloc(&d_src, bytes));
  CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), bytes, cudaMemcpyHostToDevice));

  /* ── zfp field & stream (CUDA) ───────────────────── */
  zfp_field*  field = zfp_field_3d(d_src, zfp_type_float, nx, ny, nz);
  zfp_stream* zfp   = zfp_stream_open(nullptr);
  zfp_stream_set_execution(zfp, zfp_exec_cuda);
  zfp_stream_set_accuracy(zfp, tol);

  /* 최대 압축 크기 추정 → GPU 버퍼 준비 */
  size_t maxBytes = zfp_stream_maximum_size(zfp, field);
  void*  d_cmp; CUDA_CHECK(cudaMalloc(&d_cmp, maxBytes));

  bitstream* bs = stream_open(d_cmp, maxBytes);
  zfp_stream_set_bit_stream(zfp, bs);
  zfp_stream_rewind(zfp);

  /* ── 압축 ───────────────────────────────────────── */
  size_t cmpBytes = zfp_compress(zfp, field);
  CUDA_CHECK(cudaDeviceSynchronize());

  if (cmpBytes == 0) {
    zfp_error e = zfp_stream_get_error(zfp);
    fprintf(stderr, "compress fail : %s\n", zfp_error_str(e));
    return 1;
  }
  printf("Compressed %.2f MB → %.2f MB  (%.2f×)\n",
         bytes/1e6, cmpBytes/1e6, double(bytes)/cmpBytes);

  /* ── 복원용 버퍼(GPU) 준비 ──────────────────────── */
  float* d_out; CUDA_CHECK(cudaMalloc(&d_out, bytes));
  zfp_field_set_pointer(field, d_out);

  zfp_stream_rewind(zfp);
  if (!zfp_decompress(zfp, field)) {
    zfp_error e = zfp_stream_get_error(zfp);
    fprintf(stderr, "decompress fail : %s\n", zfp_error_str(e));
    return 1;
  }

  /* ── 결과 검증 (Host) ───────────────────────────── */
  std::vector<float> h_out(n);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  double max_err = 0.0;
  for (size_t i = 0; i < n; ++i)
    max_err = fmax(max_err, fabs(h_src[i] - h_out[i]));
  printf("max |error| = %.3e (tol = %.1e)\n", max_err, tol);

  /* ── 해제 ───────────────────────────────────────── */
  stream_close(bs); zfp_stream_close(zfp); zfp_field_free(field);
  CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_cmp)); CUDA_CHECK(cudaFree(d_out));
  return 0;
}

/******************************************************************************
 * Minimal CUDA zfp example
 *  - compress & decompress on GPU
 *  - zfp 1.0.1, CUDA 11.x / 12.x
 *****************************************************************************/
#include <zfp/zfp.h>          // 기본 API
#include <zfp/errors.h>       // 오류 코드/문자열
#include <cuda_runtime.h>

#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                     \
do {                                                                         \
  cudaError_t e = (call);                                                    \
  if (e != cudaSuccess) {                                                    \
    fprintf(stderr, "CUDA %s:%d  %s\n", __FILE__, __LINE__,                  \
            cudaGetErrorString(e));                                          \
    exit(1);                                                                 \
  }                                                                          \
} while (0)

int main()
{
  /* ── 파라미터 ────────────────────────────────────────────── */
  const size_t nx = 128, ny = 128, nz = 64;          // 128×128×64
  const size_t n  = nx * ny * nz;
  const size_t bytes = n * sizeof(float);
  const double tol   = 1e-3;                         // 절대 오차 목표

  /* ── 입력 데이터 (Host) ──────────────────────────────────── */
  std::vector<float> h_src(n);
  for (size_t i = 0; i < n; ++i)
    h_src[i] = std::sin(float(i)*0.01f);

  /* ── GPU 메모리 업로드 ──────────────────────────────────── */
  float* d_src = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, bytes));
  CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), bytes, cudaMemcpyHostToDevice));

  /* ── zfp field & stream (CUDA 실행) ─────────────────────── */
  zfp_field*  field = zfp_field_3d(d_src, zfp_type_float, nx, ny, nz);
  zfp_stream* zfp   = zfp_stream_open(nullptr);

  zfp_stream_set_execution(zfp, zfp_exec_cuda);
  zfp_stream_set_accuracy(zfp, tol);                 // 손실 허용 오차

  /* 최대 압축 크기 계산 → GPU 버퍼 확보 */
  size_t maxBytes = zfp_stream_maximum_size(zfp, field);
  void*  d_cmp    = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cmp, maxBytes));

  bitstream* bs = stream_open(d_cmp, maxBytes);
  zfp_stream_set_bit_stream(zfp, bs);
  zfp_stream_rewind(zfp);

  /* ── 압축 ───────────────────────────────────────────────── */
  size_t cmpBytes = zfp_compress(zfp, field);
  CUDA_CHECK(cudaDeviceSynchronize());

  if (cmpBytes == 0) {
    fprintf(stderr, "compress error: %s\n",
            zfp_error_str(zfp_stream_get_error(zfp)));
    return 1;
  }
  printf("Compressed %.2f MB → %.2f MB  (%.2fx)\n",
         bytes/1e6, cmpBytes/1e6, double(bytes)/cmpBytes);

  /* ── 복원용 버퍼 준비 (GPU) ─────────────────────────────── */
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  zfp_field_set_pointer(field, d_out);               // 같은 field 재활용

  zfp_stream_rewind(zfp);
  if (!zfp_decompress(zfp, field)) {
    fprintf(stderr, "decompress error: %s\n",
            zfp_error_str(zfp_stream_get_error(zfp)));
    return 1;
  }

  /* ── 결과 검증 (Host) ───────────────────────────────────── */
  std::vector<float> h_out(n);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  double max_err = 0.0;
  for (size_t i = 0; i < n; ++i)
    max_err = fmax(max_err, fabs(h_src[i] - h_out[i]));
  printf("max |error| = %.3e (tol=%.1e)\n", max_err, tol);

  /* ── 메모리 해제 ────────────────────────────────────────── */
  zfp_stream_close(zfp); stream_close(bs); zfp_field_free(field);
  CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_cmp)); CUDA_CHECK(cudaFree(d_out));
  return 0;
}

#include <cstdio>
#include <vector>
#include <cmath>
#include <zfp/zfp.h>
#include <cuda_runtime.h>

int main() {
  constexpr size_t nx=64, ny=64, nz=64;
  size_t elems = nx*ny*nz;
  size_t bytes = elems * sizeof(float);

  /* ▶ 원본 데이터 */
  std::vector<float> h_src(elems);
  for(size_t i=0;i<elems;i++) h_src[i] = sinf(i*0.01f);

  float* d_src;  cudaMalloc(&d_src, bytes);
  cudaMemcpy(d_src, h_src.data(), bytes, cudaMemcpyHostToDevice);

  /* ▶ zfp 스트림 (GPU) */
  zfp_field*  field = zfp_field_3d(d_src, zfp_type_float, nx, ny, nz);
  zfp_stream* zfp   = zfp_stream_open(nullptr);
  zfp_stream_set_execution(zfp, zfp_exec_cuda);
  zfp_stream_set_accuracy(zfp, 1e-3);

  size_t maxBytes = zfp_stream_maximum_size(zfp, field);
  void*  d_compBuf; cudaMalloc(&d_compBuf, maxBytes);

  bitstream* bs = stream_open(d_compBuf, maxBytes);
  zfp_stream_set_bit_stream(zfp, bs);
  zfp_stream_rewind(zfp);

  size_t compBytes = zfp_compress(zfp, field);
  printf("compressed %zu → %zu B (%.2fx)\n",
         bytes, compBytes, double(bytes)/compBytes);

  /* ▶ GPU 비트스트림 → Host 복사 */
  std::vector<uint8_t> h_stream(compBytes);
  cudaMemcpy(h_stream.data(), d_compBuf, compBytes, cudaMemcpyDeviceToHost);

  /* ▶ CPU 복원 */
  std::vector<float> h_out(elems);
  zfp_field_set_pointer(field, h_out.data());
  zfp_stream_set_execution(zfp, zfp_exec_serial);     // serial = CPU
  stream_set_void_pointer(bs, h_stream.data(), compBytes);
  zfp_stream_rewind(zfp);

  if(!zfp_decompress(zfp, field)) {
      fprintf(stderr, "decompress fail\n");
      return 1;
  }
  printf("value[123] = %f\n", h_out[123]);

  /* ▶ 정리 */
  zfp_field_free(field);
  zfp_stream_close(zfp); stream_close(bs);
  cudaFree(d_src); cudaFree(d_compBuf);
}

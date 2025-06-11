#include <cstdio>
#include <vector>
#include <cmath>
#include <zfp/zfp.h>
#include <cuda_runtime.h>

int main() {
  //---------------- 원본 데이터 ----------------
  constexpr size_t nx=64, ny=64, nz=64;
  size_t elems = nx*ny*nz;
  std::vector<float> h_src(elems);
  for(size_t i=0;i<elems;i++) h_src[i] = std::sin(i*0.01f);

  float* d_src;
  cudaMalloc(&d_src, elems*sizeof(float));
  cudaMemcpy(d_src, h_src.data(), elems*sizeof(float), cudaMemcpyHostToDevice);

  //---------------- 압축 (GPU) ------------------
  zfp_field*  field = zfp_field_3d(d_src, zfp_type_float, nx, ny, nz);
  zfp_stream* zfp   = zfp_stream_open(nullptr);
  zfp_stream_set_execution(zfp, zfp_exec_cuda);
  zfp_stream_set_accuracy(zfp, 1e-3);

  size_t maxBytes = zfp_stream_maximum_size(zfp, field);
  void*  d_buf;  cudaMalloc(&d_buf, maxBytes);

  bitstream* bs = stream_open(d_buf, maxBytes);
  zfp_stream_set_bit_stream(zfp, bs);
  zfp_stream_rewind(zfp);

  size_t compBytes = zfp_compress(zfp, field);
  if (!compBytes) { printf("compress fail\n"); return 1; }
  printf("compressed to %zu bytes (%.2fx)\n",
         compBytes, double(elems*sizeof(float))/compBytes);

  //---------------- 복원 (CPU) ------------------
  std::vector<uint8_t> h_bitstream(compBytes);
  cudaMemcpy(h_bitstream.data(), d_buf, compBytes, cudaMemcpyDeviceToHost);

  std::vector<float> h_out(elems);
  zfp_field_set_pointer(field, h_out.data());
  zfp_stream_set_execution(zfp, zfp_exec_serial);   // CPU 모드
  stream_set_void_pointer(bs, h_bitstream.data(), compBytes);
  zfp_stream_rewind(zfp);

  if (!zfp_decompress(zfp, field)) {
    printf("decompress fail\n"); return 1;
  }
  printf("sample value = %f\n", h_out[123]);

  //---------------- 클린업 ----------------------
  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);
  cudaFree(d_src); cudaFree(d_buf);
  return 0;
}

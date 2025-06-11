#include <zfp.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>

int main() {
  constexpr size_t nx=64, ny=64, nz=64;
  size_t elems = nx*ny*nz, bytes = elems*sizeof(float);

  /* 1. 입력 데이터 → GPU */
  std::vector<float> h_src(elems);
  for(size_t i=0;i<elems;i++) h_src[i]=sinf(i*0.01f);
  float* d_src;  cudaMalloc(&d_src, bytes);
  cudaMemcpy(d_src, h_src.data(), bytes, cudaMemcpyHostToDevice);

  /* 2. zfp 설정 (GPU) */
  zfp_field*  field_gpu = zfp_field_3d(d_src, zfp_type_float,nx,ny,nz);
  zfp_stream* zfp_gpu   = zfp_stream_open(nullptr);
  zfp_stream_set_accuracy(zfp_gpu, 1e-3);
  zfp_stream_set_execution(zfp_gpu, zfp_exec_cuda);

  size_t maxBytes = zfp_stream_maximum_size(zfp_gpu, field_gpu);
  void*  d_compBuf;  cudaMalloc(&d_compBuf, maxBytes);

  bitstream* bs_gpu = stream_open(d_compBuf, maxBytes);
  zfp_stream_set_bit_stream(zfp_gpu, bs_gpu);
  zfp_stream_rewind(zfp_gpu);

  /* 3. 압축 */
  size_t compBytes = zfp_compress(zfp_gpu, field_gpu);
  cudaDeviceSynchronize();                   // ★ 필수 동기화
  if(compBytes==0){puts("compress fail");return 1;}

  printf("compressed %zu -> %zu (%.2fx)\n",bytes,compBytes,double(bytes)/compBytes);

  /* 4. GPU → Host 복사 */
  std::vector<uint8_t> h_stream(compBytes);
  cudaMemcpy(h_stream.data(), d_compBuf, compBytes, cudaMemcpyDeviceToHost);

  /* 5. zfp 설정 (CPU) */
  std::vector<float> h_out(elems);
  zfp_field* field_cpu = zfp_field_3d(h_out.data(), zfp_type_float,nx,ny,nz);
  zfp_stream* zfp_cpu  = zfp_stream_open(nullptr);
  zfp_stream_set_accuracy(zfp_cpu, 1e-3);
  zfp_stream_set_execution(zfp_cpu, zfp_exec_serial);

  bitstream* bs_cpu = stream_open(h_stream.data(), compBytes);  // ★ 새 bitstream
  zfp_stream_set_bit_stream(zfp_cpu, bs_cpu);
  zfp_stream_rewind(zfp_cpu);

  /* 6. 복원 */
  if(!zfp_decompress(zfp_cpu, field_cpu)){
      puts("decompress fail"); return 1;
  }
  printf("h_out[123] = %f\n", h_out[123]);

  /* 7. 정리 */
  stream_close(bs_gpu); zfp_stream_close(zfp_gpu);
  stream_close(bs_cpu); zfp_stream_close(zfp_cpu);
  zfp_field_free(field_gpu); zfp_field_free(field_cpu);
  cudaFree(d_src); cudaFree(d_compBuf);
}

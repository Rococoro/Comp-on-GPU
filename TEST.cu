#include <cstdio>
#include <zfp/zfp.h>
#include <cuda_runtime.h>

int main()
{
  // -------------------- 가정 --------------------
  // ➊ 원본 크기
  const size_t nx=128, ny=128, nz=128;
  size_t elems = nx*ny*nz;
  size_t bytes = elems * sizeof(float);

  // ➋ GPU 에서 이미 압축된 버퍼(d_compBuf) & 크기(compBytes) 가 존재
  //    예: 이전 단계에서 zfp_compress() 결과
  extern void*   d_compBuf;   // device pointer   (여기서는 가정)
  extern size_t  compBytes;   // 실제 압축 크기    (가정)
  //------------------------------------------------

  //---------------- 1) 압축 스트림을 Host 로 복사 ----------------
  std::vector<uint8_t> h_stream(compBytes);
  cudaMemcpy(h_stream.data(), d_compBuf, compBytes, cudaMemcpyDeviceToHost);

  //---------------- 2) zfp -field / stream 설정 ----------------
  std::vector<float> h_out(elems);             // 복원 결과 CPU 배열

  zfp_field*  field = zfp_field_3d(h_out.data(), zfp_type_float, nx, ny, nz);

  zfp_stream* zfp = zfp_stream_open(nullptr);
  zfp_stream_set_execution(zfp, zfp_exec_serial);   // ★ CPU 복원 모드

  // accuracy / rate 등은 압축할 때 썼던 것과 동일하게 맞추면 됨
  zfp_stream_set_accuracy(zfp, 1e-3);

  // bitstream → zfp 연결 (host 버퍼)
  bitstream* bs = stream_open(h_stream.data(), compBytes);
  zfp_stream_set_bit_stream(zfp, bs);
  zfp_stream_rewind(zfp);

  //---------------- 3) 복원 ----------------
  size_t out_check = zfp_decompress(zfp, field);
  if (out_check == 0) {
      fprintf(stderr, "CPU decompress failed!\n");
      return 1;
  }

  printf("CPU decompression OK, first val = %f\n", h_out[123]);

  //---------------- 4) 자원 해제 ----------------
  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);
  return 0;
}

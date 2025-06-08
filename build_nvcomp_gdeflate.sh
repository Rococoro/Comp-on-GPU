#!/usr/bin/env bash
# Jetson  / CUDA 11.4  / nvCOMP v3.x  GDeflate-only  빌드 & 설치
set -euo pipefail

# ---------- 사용자 조정 ----------
ARCH=${ARCH:-87}                 # Orin=87, Xavier NX=72
PREFIX=${PREFIX:-/opt/nvcomp}
SRC=${SRC:-$HOME/src/nvcomp}
CUDA=/usr/local/cuda-11.4        # JetPack 5.x 기본
JOBS=$(nproc)
# ---------------------------------

echo ">>> 1) 소스 가져오기"
if [ ! -d "$SRC" ]; then
  git clone --depth 1 https://github.com/NVIDIA/nvcomp.git "$SRC"
fi
cd "$SRC"

echo ">>> 2) 빌드 디렉터리 준비"
rm -rf build && mkdir build && cd build

echo ">>> 3) CMake 구성"
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCOMP_ENABLE_ZSTD=OFF \
  -DNVCOMP_ENABLE_GDEFLATE=ON \
  -DNVCOMP_ENABLE_INSTALL=ON \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA" \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH"

echo ">>> 4) 컴파일 ($JOBS 병렬)"
make -j"$JOBS"

echo ">>> 5) 설치 → $PREFIX"
sudo make install

echo ">>> 6) 환경변수(세션 한정) 설정"
export nvcomp_DIR=$PREFIX/lib/cmake/nvcomp
export CMAKE_PREFIX_PATH=$PREFIX:$CMAKE_PREFIX_PATH
echo "        nvcomp_DIR     = $nvcomp_DIR"
echo "        CMAKE_PREFIX_PATH += $PREFIX"
echo "<<< 완료"

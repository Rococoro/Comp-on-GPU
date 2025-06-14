cmake_minimum_required(VERSION 3.18)
project(cuda_zfp_demo LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 87)      # Orin: 87 / Xavier NX: 72

list(APPEND CMAKE_PREFIX_PATH "/opt/zfp")   # zfp 설치 경로
find_package(zfp CONFIG REQUIRED)

add_executable(cuda_zfp_test cuda_zfp_test.cu)
target_link_libraries(cuda_zfp_test PRIVATE zfp::zfp)   # 1.0.1은 단일 타깃

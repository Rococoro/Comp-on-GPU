nvidia@tegra-ubuntu:~/Downloads/nvcomp/nvcomp/build$ ./cmake.sh 
Release build.
-- Finding CUB
Build nvCOMP version 2.2.0
-- Configuring done (0.1s)
-- Generating done (0.0s)
-- Build files have been written to: /home/nvidia/Downloads/nvcomp/nvcomp/build
Release build.
-- Finding CUB
Build nvCOMP version 2.2.0
CMake Error: File /home/nvidia/Downloads/nvcomp/nvcomp/cmake/nvcomp-config.cmake.in does not exist.
CMake Error at /usr/local/share/cmake-3.31/Modules/CMakePackageConfigHelpers.cmake:507 (configure_file):
  configure_file Problem configuring file
Call Stack (most recent call first):
  CMakeLists.txt:171 (configure_package_config_file)


CMake Error: File /home/nvidia/Downloads/nvcomp/nvcomp/cmake/nvcomp-config.cmake.in does not exist.
CMake Error at /usr/local/share/cmake-3.31/Modules/CMakePackageConfigHelpers.cmake:507 (configure_file):
  configure_file Problem configuring file
Call Stack (most recent call first):
  CMakeLists.txt:181 (configure_package_config_file)


-- Configuring incomplete, errors occurred!
make: *** [Makefile:201: cmake_check_build_system] Error 1
[  3%] Building CUDA object src/CMakeFiles/nvcomp.dir/BitPackGPU.cu.o
[  7%] Building CUDA object src/CMakeFiles/nvcomp.dir/CudaUtils.cu.o
[ 10%] Building CUDA object src/CMakeFiles/nvcomp.dir/DeltaGPU.cu.o
[ 14%] Building CUDA object src/CMakeFiles/nvcomp.dir/RunLengthEncodeGPU.cu.o
^Cnvcc error   : 'cicc' died due to signal 2 
make[2]: *** [src/CMakeFiles/nvcomp.dir/build.make:125: src/CMakeFiles/nvcomp.dir/RunLengthEncodeGPU.cu.o] Interrupt
make[1]: *** [CMakeFiles/Makefile2:106: src/CMakeFiles/nvcomp.dir/all] Interrupt
make: *** [Makefile:136: all] Interrupt
nvidia@tegra-ubuntu:~/Downloads/nvcomp/nvcomp/build$ ./cmake.sh 
Release build.
-- Finding CUB
Build nvCOMP version 2.2.0
-- Configuring done (0.1s)
-- Generating done (0.0s)
-- Build files have been written to: /home/nvidia/Downloads/nvcomp/nvcomp/build
Release build.
-- Finding CUB
Build nvCOMP version 2.2.0
CMake Error: File /home/nvidia/Downloads/nvcomp/nvcomp/cmake/nvcomp-config.cmake.in does not exist.
CMake Error at /usr/local/share/cmake-3.31/Modules/CMakePackageConfigHelpers.cmake:507 (configure_file):
  configure_file Problem configuring file
Call Stack (most recent call first):
  CMakeLists.txt:171 (configure_package_config_file)


CMake Error: File /home/nvidia/Downloads/nvcomp/nvcomp/cmake/nvcomp-config.cmake.in does not exist.
CMake Error at /usr/local/share/cmake-3.31/Modules/CMakePackageConfigHelpers.cmake:507 (configure_file):
  configure_file Problem configuring file
Call Stack (most recent call first):
  CMakeLists.txt:181 (configure_package_config_file)


-- Configuring incomplete, errors occurred!
make: *** [Makefile:201: cmake_check_build_system] Error 1
[  3%] Building CUDA object src/CMakeFiles/nvcomp.dir/RunLengthEncodeGPU.cu.o

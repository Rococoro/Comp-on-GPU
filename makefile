# ────────────────────────────────────────────────────────────────
#  Simple Makefile for Comp-on-GPU project
#    • src/*.cu      → build/*.o
#    • tests/*.cu    → bin/test_*.exe
#  CUDA 12.x, nvCOMP, zstd
# ────────────────────────────────────────────────────────────────

########################
#  toolchain & flags   #
########################
CUDA_PATH ?= /usr/local/cuda
NVCC      := $(CUDA_PATH)/bin/nvcc
ARCH      ?= sm_75           # 변경 시: make ARCH=sm_86 …

CFLAGS := -O3 -std=c++17 \
          -gencode arch=compute_$(ARCH:sm_%=%),code=sm_$(ARCH:sm_%=%) \
          -Xptxas=-dlcm=ca,-maxrregcount=64
NVCOMP_DIR ?= /usr/include/nvcomp

INC    := -Iinclude -I$(NVCOMP_DIR)
LIBS   := -lnvcomp -lzstd

########################
#  source / objects    #
########################
SRC   := $(wildcard src/*.cu)
OBJ   := $(patsubst src/%.cu,build/%.o,$(SRC))

TEST_SRC := tests/test_files.cu tests/test_pipe1.cu
TEST_EXE := $(patsubst tests/%.cu,bin/%,$(TEST_SRC))

########################
#  default target      #
########################
.PHONY: all
all: $(TEST_EXE)

########################
#  build rules         #
########################
# object files
build/%.o: src/%.cu | build
	$(NVCC) $(CFLAGS) $(INC) -dc $< -o $@

# link each test executable with all kernels/objs
bin/%: tests/%.cu $(OBJ) | bin
	$(NVCC) $(CFLAGS) $(INC) $< $(OBJ) $(LIBS) -o $@

########################
#  utility targets     #
########################
build:
	mkdir -p build

bin:
	mkdir -p bin

.PHONY: clean
clean:
	rm -rf build bin

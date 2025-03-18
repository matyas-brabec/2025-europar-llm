CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
CXX = g++

CXXFLAGS = -std=c++20 -Wall -Wextra -pedantic -O3
NVCCFLAGS = -ccbin $(CXX) -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 -std=c++20 -Xcompiler -Wall,-Wextra -O3 -I. --extended-lambda
LDFLAGS = -lcudart

LIB_DIRS = -L$(CUDA_PATH)/lib64
INCLUDE_DIRS = -I$(CUDA_PATH)/include

GOL_IMPL ?= gol.cu
GOL_IMPL_DIR = $(dir $(GOL_IMPL))

BIN_DIR = bin
OBJ_DIR = $(BIN_DIR)/obj
EXECUTABLE = $(BIN_DIR)/game_of_life

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJ_DIR)/main.o $(OBJ_DIR)/gol.o
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LIB_DIRS) $(INCLUDE_DIRS)

$(OBJ_DIR)/main.o: main.cpp gol.cuh
	mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LDFLAGS) $(LIB_DIRS) $(INCLUDE_DIRS)

$(OBJ_DIR)/gol.o: $(GOL_IMPL) gol.cuh
	mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(BIN_DIR)

run: $(EXECUTABLE)
	@python run-code.py $(GOL_IMPL)

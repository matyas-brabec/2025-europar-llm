CPP=g++
STD=-std=c++20
CFLAGS=-Wall -O3 $(STD)
NVCCFLAGS=-ccbin $(CPP) $(STD) -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90
INCLUDE=/usr/local/cuda/include ./bpplib
LDFLAGS=
LIBS=
LIBDIRS=/usr/local/cuda/lib64
HEADERS=$(shell find . -name '*.hpp')
KERNELS=$(shell find . -maxdepth 1 -name '*.cu')
KERNEL_OBJS=$(patsubst %.cu,%.obj,$(KERNELS))
TARGET=histogram

DATA ?= txt

.PHONY: all clear clean purge run

all: $(TARGET)



# Building Targets

$(TARGET): $(TARGET).cpp $(HEADERS) $(KERNEL_OBJS)
	@echo Compiling and linking executable "$@" ...
	@$(CPP) $(CFLAGS) $(addprefix -I,$(INCLUDE)) $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(KERNEL_OBJS) $< -o $@ $(addprefix -l,$(LIBS)) -lcudart

%.obj: %.cu
	@echo Compiling kernels in "$<" ...
	@nvcc $(NVCCFLAGS) $(addprefix -I,$(INCLUDE)) --compile -cudart static $< -o $@



run:
	@./histogram --algorithm cuda --verify --warmup --fromValue 32 --toValue 127 --repeatInput 16k ./data/lorem_small.$(DATA)

# Cleaning Stuff

clear:
	@echo Removing object files ...
	-@rm -f *.obj

clean: clear

purge: clear
	@echo Removing executable ...
	-@rm -f $(TARGET)

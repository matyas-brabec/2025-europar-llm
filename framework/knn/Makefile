#!/bin/env make

BUILD_DIR = build
SOURCE_DIR = src
INCLUDE_DIR = include

BUILD_TYPE = Release

N = 4194304
K = 1024
M = 4096
S = 24
R = 10
W = 3

CMAKE_COMMAND = cmake
CMAKE_FLAGS = -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	-DCMAKE_CUDA_ARCHITECTURES="70;80;86;89;90" \
	-DCMAKE_CUDA_HOST_COMPILER="${CXX}"

.PHONY: all clean compile run configure help head

all: ${BUILD_DIR}/knn

help:
	@"${BUILD_DIR}/knn" --help

head:
	@"${BUILD_DIR}/knn" --head

run:
	@timeout 2m \
	"${BUILD_DIR}/knn" -n ${N} -k ${K} -m ${M} -s ${S} -r ${R} -w ${W}

${BUILD_DIR}/knn: compile

compile: ${BUILD_DIR}/CMakeCache.txt
	${CMAKE_COMMAND} --build "${BUILD_DIR}" --config "${BUILD_TYPE}" --parallel 4

${BUILD_DIR}/CMakeCache.txt:
	${CMAKE_COMMAND} -S . -B "${BUILD_DIR}" ${CMAKE_FLAGS}

clean:
	rm -rf "${BUILD_DIR}"

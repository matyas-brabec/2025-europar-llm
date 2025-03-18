cmake_minimum_required(VERSION 3.20)

project(knn LANGUAGES CXX CUDA)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler -O3")

FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG        v3.2.0)
FetchContent_MakeAvailable(cxxopts)

find_package(CUDAToolkit REQUIRED)

include_directories(include)
include_directories(${CMAKE_SOURCE_DIR})

add_executable(knn src/main.cpp src/knn.cu src/baseline.cu)

target_link_libraries(knn PRIVATE cxxopts::cxxopts)
target_link_libraries(knn PRIVATE CUDA::cudart)

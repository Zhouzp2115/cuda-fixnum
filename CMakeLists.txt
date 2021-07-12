cmake_minimum_required(VERSION 3.10)
project(paillier LANGUAGES CXX CUDA)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-std=c++11;)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

#lib dir
link_directories("./bin")

find_package(CUDA REQUIRED)

#build paillier lib
include_directories("./src")
cuda_add_library(paillier SHARED "paillier/paillier.cu")

#test paillier lib
add_executable(test_pailler "paillier/test_pailler.cpp")
target_link_libraries(test_pailler "paillier")


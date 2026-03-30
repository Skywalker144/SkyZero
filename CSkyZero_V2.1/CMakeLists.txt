cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(test-libtorch)

set(TORCH_CUDA_ARCH_LIST "8.9")

find_package(Torch REQUIRED)
set(CMAKE_CXX_STANDARD 17)

add_executable(test-libtorch gomoku_train.cpp)

target_link_libraries(test-libtorch "${TORCH_LIBRARIES}")
set_property(TARGET test-libtorch PROPERTY CXX_STANDARD 17)

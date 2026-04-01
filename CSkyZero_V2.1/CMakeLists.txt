cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(gomoku_train)

set(TORCH_CUDA_ARCH_LIST "8.9")

find_package(Torch REQUIRED)
set(CMAKE_CXX_STANDARD 17)

add_executable(gomoku_train gomoku/gomoku_train.cpp)
target_link_libraries(gomoku_train "${TORCH_LIBRARIES}")

add_executable(gomoku_play gomoku/gomoku_play.cpp)
target_link_libraries(gomoku_play "${TORCH_LIBRARIES}")

set_property(TARGET gomoku_train PROPERTY CXX_STANDARD 17)

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(skyzero_selfplay)

set(CMAKE_CXX_STANDARD 17)
set(TORCH_CUDA_ARCH_LIST "8.9")

find_package(Torch REQUIRED)

# Find libzip
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(LIBZIP QUIET libzip)
endif()

if(NOT LIBZIP_FOUND)
    # Search in conda environment and standard paths
    find_path(LIBZIP_INCLUDE_DIR zip.h
        HINTS $ENV{CONDA_PREFIX}/include /usr/include /usr/local/include)
    find_library(LIBZIP_LIBRARY zip
        HINTS $ENV{CONDA_PREFIX}/lib /usr/lib /usr/local/lib)

    if(LIBZIP_INCLUDE_DIR AND LIBZIP_LIBRARY)
        set(LIBZIP_INCLUDE_DIRS ${LIBZIP_INCLUDE_DIR})
        set(LIBZIP_LIBRARIES ${LIBZIP_LIBRARY})
        message(STATUS "Found libzip: ${LIBZIP_LIBRARY}")
    else()
        message(FATAL_ERROR "libzip not found. Install via: conda install libzip  OR  apt install libzip-dev")
    endif()
endif()

add_executable(selfplay
    selfplay_main.cpp
    numpywrite.cpp
)

target_include_directories(selfplay PRIVATE ${LIBZIP_INCLUDE_DIRS})
target_link_libraries(selfplay "${TORCH_LIBRARIES}" ${LIBZIP_LIBRARIES})
set_property(TARGET selfplay PROPERTY CXX_STANDARD 17)

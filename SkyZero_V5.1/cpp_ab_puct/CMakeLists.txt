cmake_minimum_required(VERSION 3.18)
project(skyzero_v5_ab_puct CXX)

# Throwaway sibling of SkyZero_V5/cpp/ that builds ONLY gomoku_ab with an
# added PUCT search path (puct_search() in alphazero_parallel.h). Loads V5
# model weights as-is — V5/V6 networks differ but this binary uses V5's own
# nets, so weight compatibility is automatic. Build dir lives here under
# cpp_ab_puct/build/, completely isolated from cpp/build/.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- MAX_BOARD_SIZE from V5 scripts/run.cfg ------------------------------
# Mirror cpp/CMakeLists.txt: pull MAX_BOARD_SIZE from the SAME run.cfg as V5
# so this binary stays compatible with V5's exported model.
set(_run_cfg "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/run.cfg")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_run_cfg}")
if(EXISTS "${_run_cfg}")
    file(STRINGS "${_run_cfg}" _max_lines REGEX "^[ \t]*MAX_BOARD_SIZE[ \t]*=")
    if(_max_lines)
        list(GET _max_lines 0 _max_line)
        string(REGEX REPLACE "^[ \t]*MAX_BOARD_SIZE[ \t]*=[ \t]*([0-9]+).*" "\\1"
               SKYZERO_MAX_BOARD_SIZE "${_max_line}")
    endif()
endif()
if(NOT SKYZERO_MAX_BOARD_SIZE)
    set(SKYZERO_MAX_BOARD_SIZE 15)
    message(STATUS "SKYZERO_MAX_BOARD_SIZE = ${SKYZERO_MAX_BOARD_SIZE} (default; run.cfg has no MAX_BOARD_SIZE=)")
else()
    message(STATUS "SKYZERO_MAX_BOARD_SIZE = ${SKYZERO_MAX_BOARD_SIZE} (from V5 scripts/run.cfg)")
endif()
add_compile_definitions(SKYZERO_MAX_BOARD_SIZE=${SKYZERO_MAX_BOARD_SIZE})

# --- LibTorch -------------------------------------------------------------
find_package(Torch REQUIRED)

# --- Threads -------------------------------------------------------------
find_package(Threads REQUIRED)

# --- gomoku_ab_puct binary -----------------------------------------------
# Renamed from `gomoku_ab` so a stray PATH lookup can't pick this up in place
# of the production V5 binary at cpp/build/gomoku_ab.
add_executable(gomoku_ab_puct gomoku_ab_main.cpp)
target_include_directories(gomoku_ab_puct PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gomoku_ab_puct PRIVATE
    ${TORCH_LIBRARIES}
    Threads::Threads
)
target_compile_options(gomoku_ab_puct PRIVATE -Wall -Wextra -Wno-unused-parameter)
set_property(TARGET gomoku_ab_puct PROPERTY CXX_EXTENSIONS OFF)

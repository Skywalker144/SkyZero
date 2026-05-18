cmake_minimum_required(VERSION 3.18)
project(skyzero_v4 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- MAX_BOARD_SIZE from scripts/run.cfg ---------------------------------
# Auto-pull MAX_BOARD_SIZE=<n> from run.cfg and pass as -DSKYZERO_MAX_BOARD_SIZE=<n>
# to all targets (consumed by cpp/envs/gomoku.h). Pattern follows KataGomo's
# COMPILE_MAX_BOARD_LEN, but the value comes from the project's single config
# file instead of a separate CMake option, so editing run.cfg is enough — no
# need to touch gomoku.h or pass -D manually.
#
# CMAKE_CONFIGURE_DEPENDS makes cmake re-run when run.cfg changes; if the value
# changed, gomoku.h's effective MAX_BOARD_SIZE changes, and ninja/make rebuilds
# every .cpp that includes it.
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
    message(STATUS "SKYZERO_MAX_BOARD_SIZE = ${SKYZERO_MAX_BOARD_SIZE} (from scripts/run.cfg)")
endif()
add_compile_definitions(SKYZERO_MAX_BOARD_SIZE=${SKYZERO_MAX_BOARD_SIZE})

# --- LibTorch -------------------------------------------------------------
# Expect caller to set CMAKE_PREFIX_PATH to LibTorch install dir, e.g.:
#   cmake -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch
find_package(Torch REQUIRED)

# --- libzip --------------------------------------------------------------
find_package(libzip CONFIG QUIET)
if(NOT libzip_FOUND)
    # Fall back to pkg-config / manual find.
    find_path(LIBZIP_INCLUDE_DIR zip.h)
    find_library(LIBZIP_LIBRARY NAMES zip)
    if(NOT LIBZIP_INCLUDE_DIR OR NOT LIBZIP_LIBRARY)
        message(FATAL_ERROR "libzip not found. Install libzip-dev.")
    endif()
    add_library(libzip::zip INTERFACE IMPORTED)
    set_target_properties(libzip::zip PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LIBZIP_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${LIBZIP_LIBRARY}"
    )
endif()

# --- Threads -------------------------------------------------------------
find_package(Threads REQUIRED)

# --- selfplay binary -----------------------------------------------------
add_executable(selfplay_main selfplay_main.cpp)
target_include_directories(selfplay_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(selfplay_main PRIVATE
    ${TORCH_LIBRARIES}
    libzip::zip
    Threads::Threads
)
target_compile_options(selfplay_main PRIVATE -Wall -Wextra -Wno-unused-parameter)

# Some LibTorch distributions require an extra ABI flag at link time.
set_property(TARGET selfplay_main PROPERTY CXX_EXTENSIONS OFF)

# --- mcts_probe binary ---------------------------------------------------
# Post-export diagnostic: runs one MCTS search on an empty board and prints
# the root value. Shares AlphaZeroConfig parsing conventions with selfplay.
add_executable(mcts_probe mcts_probe_main.cpp)
target_include_directories(mcts_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mcts_probe PRIVATE
    ${TORCH_LIBRARIES}
    Threads::Threads
)
target_compile_options(mcts_probe PRIVATE -Wall -Wextra -Wno-unused-parameter)
set_property(TARGET mcts_probe PROPERTY CXX_EXTENSIONS OFF)

# --- gomoku_play binary --------------------------------------------------
# Interactive human-vs-AI Gomoku. Loads the same TorchScript model as
# selfplay/probe but reads scripts/play.cfg for its own MCTS knobs.
add_executable(gomoku_play gomoku_play_main.cpp)
target_include_directories(gomoku_play PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gomoku_play PRIVATE
    ${TORCH_LIBRARIES}
    Threads::Threads
)
target_compile_options(gomoku_play PRIVATE -Wall -Wextra -Wno-unused-parameter)
set_property(TARGET gomoku_play PROPERTY CXX_EXTENSIONS OFF)

# --- gomoku_elo binary ---------------------------------------------------
# Headless model-A-vs-model-B match runner. Appends per-game JSONL results
# for python/elo.py to consume.
add_executable(gomoku_elo gomoku_elo_main.cpp)
target_include_directories(gomoku_elo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gomoku_elo PRIVATE
    ${TORCH_LIBRARIES}
    Threads::Threads
)
target_compile_options(gomoku_elo PRIVATE -Wall -Wextra -Wno-unused-parameter)
set_property(TARGET gomoku_elo PROPERTY CXX_EXTENSIONS OFF)

# --- gomoku_ab binary ----------------------------------------------------
# Headless A-vs-B MCTS-config match runner. Plays the same model against
# itself with two different search configs (scripts/ab/a.cfg vs b.cfg) and
# appends per-game JSONL results for python/ab.py to consume.
add_executable(gomoku_ab gomoku_ab_main.cpp)
target_include_directories(gomoku_ab PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gomoku_ab PRIVATE
    ${TORCH_LIBRARIES}
    Threads::Threads
)
target_compile_options(gomoku_ab PRIVATE -Wall -Wextra -Wno-unused-parameter)
set_property(TARGET gomoku_ab PROPERTY CXX_EXTENSIONS OFF)

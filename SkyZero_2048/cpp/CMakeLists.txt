cmake_minimum_required(VERSION 3.18)
project(skyzero_2048 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- LibTorch -------------------------------------------------------------
# Caller sets CMAKE_PREFIX_PATH to PyTorch's bundled LibTorch (see scripts/
# env_paths.cfg's LIBTORCH). Using the same LibTorch as Python guarantees ABI.
find_package(Torch REQUIRED)

# --- libzip (npz output for self-play) -----------------------------------
find_package(libzip CONFIG QUIET)
if(NOT libzip_FOUND)
    find_path(LIBZIP_INCLUDE_DIR zip.h)
    find_library(LIBZIP_LIBRARY NAMES zip)
    if(NOT LIBZIP_INCLUDE_DIR OR NOT LIBZIP_LIBRARY)
        message(FATAL_ERROR "libzip not found. Install libzip-dev.")
    endif()
    add_library(libzip::zip INTERFACE IMPORTED)
    set_target_properties(libzip::zip PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LIBZIP_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${LIBZIP_LIBRARY}")
endif()

find_package(Threads REQUIRED)

# --- selfplay2048: single-thread bridge / eval binary --------------------
# Loads a TorchScript Net2048, runs the afterstate Gumbel MCTS
# (skyzero_2048.h + envs/game2048.h). No libzip (no npz output).
add_executable(selfplay2048 selfplay2048_main.cpp)
target_include_directories(selfplay2048 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(selfplay2048 PRIVATE ${TORCH_LIBRARIES} Threads::Threads)
target_compile_options(selfplay2048 PRIVATE -Wall -Wextra -Wno-unused-parameter)
set_property(TARGET selfplay2048 PROPERTY CXX_EXTENSIONS OFF)

# --- selfplay2048_par: parallel self-play (the throughput path) ----------
# Worker threads + central batched inference server (infer_server_2048.h),
# writes npz shards (npy.h) that az2048/train.py consumes.
add_executable(selfplay2048_par selfplay2048_par_main.cpp)
target_include_directories(selfplay2048_par PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(selfplay2048_par PRIVATE ${TORCH_LIBRARIES} libzip::zip Threads::Threads)
target_compile_options(selfplay2048_par PRIVATE -Wall -Wextra -Wno-unused-parameter)
set_property(TARGET selfplay2048_par PROPERTY CXX_EXTENSIONS OFF)

# --- mcts_probe_2048: empty-board root-value diagnostic -> probe.tsv ------
add_executable(mcts_probe_2048 mcts_probe_2048_main.cpp)
target_include_directories(mcts_probe_2048 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mcts_probe_2048 PRIVATE ${TORCH_LIBRARIES} Threads::Threads)
target_compile_options(mcts_probe_2048 PRIVATE -Wall -Wextra -Wno-unused-parameter)
set_property(TARGET mcts_probe_2048 PROPERTY CXX_EXTENSIONS OFF)

# --- reload_stress_2048: InferenceServer2048::reload() concurrency gate ---
add_executable(reload_stress_2048 reload_stress_2048_main.cpp)
target_include_directories(reload_stress_2048 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(reload_stress_2048 PRIVATE ${TORCH_LIBRARIES} Threads::Threads)
target_compile_options(reload_stress_2048 PRIVATE -Wall -Wextra -Wno-unused-parameter)
set_property(TARGET reload_stress_2048 PROPERTY CXX_EXTENSIONS OFF)

cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)
project(bench-solver CXX)

add_executable(${PROJECT_NAME} bench.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${TOPDIR}/src)
target_link_libraries(${PROJECT_NAME} PRIVATE solve)
target_compile_options(${PROJECT_NAME} PRIVATE -march=${TARGET_ARCH})

# Optional: cold-cache mode (flushes cache between iterations)
# cmake -DCOLD_CACHE=ON ...
option(COLD_CACHE "Flush cache between benchmark iterations" OFF)
if(COLD_CACHE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE COLD_CACHE)
endif()

# Optional: per-section timing (instruments solve.cpp internals)
# cmake -DSOLVE_TIMING=ON ...
option(SOLVE_TIMING "Print per-section timing breakdown" OFF)
if(SOLVE_TIMING)
    target_compile_definitions(${PROJECT_NAME} PRIVATE SOLVE_TIMING)
    target_compile_definitions(solve PRIVATE SOLVE_TIMING)
endif()

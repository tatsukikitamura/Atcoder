cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.27.0")
    cmake_policy(SET CMP0144 NEW)
endif()

project(run-solver CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")

SET(TOPDIR ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH native)
endif()

add_executable(${PROJECT_NAME} src/main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${TOPDIR}/src)
target_link_libraries(${PROJECT_NAME} PRIVATE solve)

add_subdirectory(src)
add_subdirectory(bench)

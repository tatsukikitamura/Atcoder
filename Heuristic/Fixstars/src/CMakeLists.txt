cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.27.0")
    cmake_policy(SET CMP0144 NEW)
endif()

project(solve CXX)

add_library(${PROJECT_NAME} SHARED solve.cpp)

target_compile_options(${PROJECT_NAME} PRIVATE -march=${TARGET_ARCH})

target_include_directories(${PROJECT_NAME} PRIVATE ${TOPDIR}/src)

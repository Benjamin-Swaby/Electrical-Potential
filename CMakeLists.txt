cmake_minimum_required(VERSION 3.20)

enable_language(CUDA)


set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS  "-O2 -Wall -Wpedantic")

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

project(EPot
    LANGUAGES CUDA
)


add_executable(epot
    src/main.cu
    kernel/kernel.cu
)

target_include_directories(epot
    PRIVATE
    kernel/
    include/
)
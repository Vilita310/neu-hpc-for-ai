cmake_minimum_required(VERSION 3.18)
project(flashattn_week4 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(flashattn
    src/flashattn_cpu.cpp
    src/flashattn_cuda.cu
)

target_include_directories(flashattn PUBLIC include)

set_target_properties(flashattn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(flashattn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

add_executable(flashattn_test tests/test_main.cpp)
target_link_libraries(flashattn_test PRIVATE flashattn)

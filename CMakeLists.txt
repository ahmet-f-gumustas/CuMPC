cmake_minimum_required(VERSION 3.24)
project(cumpc_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 4070 (Ada Lovelace) — Phase 0 kararı; manuel build'de -DCMAKE_CUDA_ARCHITECTURES=89
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(cumpc_core
  bindings/pybind_module.cpp
  core/sampling.cu
  core/rollout.cu
  core/reduction.cu
  core/mppi.cu
  core/debug.cu
  perception/costmap.cpp
  perception/elevation.cpp
)

target_include_directories(cumpc_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cumpc_core PRIVATE CUDA::cudart)
target_compile_options(cumpc_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --Werror=all-warnings>
)

install(TARGETS cumpc_core LIBRARY DESTINATION .)

cmake_minimum_required(VERSION 3.15)

project(rocm-hacking VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(vendor/rocm_systems/projects/rocm-smi-lib)
include_directories(vendor/rocm_systems/projects/rocm-smi-lib/include)
include_directories(vendor/rocm_systems/projects/rocm-smi-lib/third_party/shared_mutex)

add_executable(rocm_app rocm_hacking.cc)

target_link_libraries(rocm_app PRIVATE rocm_smi64)
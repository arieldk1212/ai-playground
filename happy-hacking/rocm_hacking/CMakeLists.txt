cmake_minimum_required(VERSION 3.15)

project(rocm-hacking VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(rocm_smi_lib)
include_directories(rocm_smi_lib/include)
include_directories(rocm_smi_lib/third_party/shared_mutex)

add_executable(rocm_app rocm_hacking.cc)

target_link_libraries(rocm_app PRIVATE rocm_smi64)
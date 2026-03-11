cmake_minimum_required(VERSION 3.15)
project(my_kmeans_project LANGUAGES CXX)

# 1. Set C++ standard (C++17 is recommended for nanobind)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Find nanobind and Python
find_package(nanobind CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# 3. Define the Python module
# This creates "my_kmeans.so" or "my_kmeans.pyd"
nanobind_add_module(
    my_kmeans           # Module name used in 'import my_kmeans'
    NB_STATIC           # Recommended for simple cloud deployments
    src/kmeans_module.cpp
)

# 4. Optional: Enable OpenMP for Cloud Parallelism
# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#     target_link_libraries(my_kmeans PRIVATE OpenMP::OpenMP_CXX)
# endif()
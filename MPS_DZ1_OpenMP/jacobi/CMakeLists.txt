cmake_minimum_required(VERSION 3.9)

set(CMAKE_C_COMPILER "/usr/local/bin/gcc-8" CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-8" CACHE STRING "C++ compiler" FORCE)

project(ia130010d_jacobi)

find_package(OpenMP)

add_executable(jacobi jacobi.cc)

if(OpenMP_CXX_FOUND)
    target_link_libraries(jacobi PUBLIC OpenMP::OpenMP_CXX)
endif()
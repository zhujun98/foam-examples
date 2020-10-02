cmake_minimum_required(VERSION 3.13)
project(image_process)

set(CMAKE_CXX_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(foamalgo REQUIRED)
if(foamalgo_FOUND)
    message(STATUS "Found foamalgo version: ${foamalgo_INCLUDE_DIR}")
endif()

set(TBB_INCLUDE_DIR ${CMAKE_PREFIX_PATH}/include)
set(TBB_LIBRARY ${CMAKE_PREFIX_PATH}/lib/libtbb.so)

add_executable(run main.cpp)
target_include_directories(run PRIVATE ${TBB_INCLUDE_DIR})
target_link_libraries(run PRIVATE foamalgo ${TBB_LIBRARY})

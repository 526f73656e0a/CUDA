# CMakeLists.txt
cmake_minimum_required(VERSION 2.8)
project( tut7)

# set( CUDA_TOOLKIT_ROOT_DIR "/Developer/NVIDIA/CUDA-9.0/")
find_package(CUDA QUIET REQUIRED)

set( ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set( SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(CUDA_SOURCE
   ${SRC_DIR}/main.cu
)

IF (WIN32)
ELSE()
  set( CMAKE_CXX_FLAGS "-g -W -Wall -std=c++11")
ENDIF()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

include_directories( ${INCLUDE_DIRS})

# set(CMAKE_BUILD_TYPE Debug)

cuda_add_executable( main ${CUDA_SOURCE})

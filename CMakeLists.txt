cmake_minimum_required(VERSION 3.15)
project(inference_engine CXX)

set(CMAKE_C_FLAGS_DEBUG "-g3 -O3 -Wall -Wextra -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra")

set(default_build_type "Debug")
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}")
endif()

set(CMAKE_CXX_STANDARD 14)

## setup onnx
find_package(Git)
if(NOT GIT_FOUND)
  message(FATAL_ERROR "git not found")
endif()

message(STATUS "Adding external/onnx")
set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
set(ONNX_SRC_DIR ${EXTERNAL_DIR}/onnx)
set(CATCH2_SRC_DIR ${EXTERNAL_DIR}/Catch2)

execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init -- ${ONNX_SRC_DIR} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init -- ${CATCH2_SRC_DIR} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set(ONNX_ML 0)
add_subdirectory(${EXTERNAL_DIR}/onnx)
add_subdirectory(${EXTERNAL_DIR}/Catch2)

add_subdirectory(inference_engine)
add_subdirectory(example)
add_subdirectory(test)

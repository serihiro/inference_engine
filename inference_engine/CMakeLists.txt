find_package(OpenCV)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV is not found. OpenCV is needed to build `image_util.cpp`.")
endif()

find_package(Protobuf ${PROTOBUF_VERSION} REQUIRED)
if (NOT Protobuf_FOUND)
    message(FATAL_ERROR "Protobuf is not found. Protobuf is needed to build `onnx.cpp`.")
endif()

add_library(
  inference_engine_lib 
    OBJECT
      image_util.cpp
      inferer.cpp
      naive_backend.cpp
      onnx.cpp
)

target_include_directories(inference_engine_lib
  PUBLIC
    $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${Protobuf_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/onnx>
    $<BUILD_INTERFACE:${ONNX_INCLUDE_DIRS}>
)

target_link_directories(inference_engine_lib
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/onnx>
)

link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${Protobuf_LIBRARY_DIRS})

target_link_libraries(
  inference_engine_lib
    PUBLIC
      onnx
      onnx_proto
      "${OpenCV_LIBRARIES}"
      "${Protobuf_LIBRARIES}"
)

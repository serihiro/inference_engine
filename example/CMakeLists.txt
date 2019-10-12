find_package(OpenCV)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV is not found. OpenCV is needed to build `image_util.cpp`.")
endif()

find_package(Protobuf ${PROTOBUF_VERSION} REQUIRED)
if (NOT Protobuf_FOUND)
    message(FATAL_ERROR "Protobuf is not found. Protobuf is needed to build `onnx.cpp`.")
endif()

target_include_directories(inference_engine_lib
  PUBLIC
    $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${Protobuf_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/onnx>
    $<BUILD_INTERFACE:${ONNX_INCLUDE_DIRS}>
)

add_executable(imagenet_vgg19.o imagenet_vgg19.cpp)
target_link_libraries(imagenet_vgg19.o
  PUBLIC
    inference_engine_lib
)

add_executable(mnist_mlp.o mnist_mlp.cpp)
target_link_libraries(mnist_mlp.o
  PUBLIC
    inference_engine_lib
)

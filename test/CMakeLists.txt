find_package(OpenCV)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV is not found. OpenCV is needed to build `test_image_util.cpp`.")
endif()

add_executable(test_image_util.o test_image_util.cpp util.cpp)
target_link_libraries(test_image_util.o
  PUBLIC
    $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
    inference_engine_lib
    Catch2::Catch2
)

add_executable(test_naive_backend.o test_naive_backend.cpp util.cpp)
target_link_libraries(test_naive_backend.o
  PUBLIC
    inference_engine_lib
    Catch2::Catch2
)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/gray.jpg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

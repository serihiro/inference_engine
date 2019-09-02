CC = clang++
CFLAGS = -std=c++14 -g3 -O2 -Wall -Wextra `pkg-config --cflags protobuf` `pkg-config --cflags opencv4`
INCLUDE = `pkg-config --libs protobuf` `pkg-config --libs opencv4`
SRCS=$(wildcard *.cpp)
OBJS=$(SRCS:.cpp=.o)
HEADERS=$(wildcard *.hpp)


image_util.o: image_util.cpp
	$(CC) $^ $(CFLAGS) -o $@ -c

naive_backend.o: naive_backend.cpp
	$(CC) $^ $(CFLAGS) -o $@ -c

onnx.o: onnx.cpp
	$(CC) $^ $(CFLAGS) -o $@ -c

onnx_client.o: ../external/onnx/onnx/onnx.pb.cc
	$(CC) $^ $(CFLAGS) -o $@ -c

../external/onnx/onnx/onnx.pb.cc: ../external/onnx/onnx/onnx.proto
	protoc -I=../external/onnx/onnx --cpp_out=../external/onnx/onnx ../external/onnx/onnx/onnx.proto

.PHONY :format
format:
	clang-format $(SRCS) $(HEADERS) -i

.PHONY :clean
clean:
	$(RM) -rf ./*.o

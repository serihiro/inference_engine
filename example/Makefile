CC = clang++
CFLAGS = -std=c++14 -g3 -O2 -Wall -fsanitize=address -Wextra `pkg-config --cflags protobuf` `pkg-config --cflags opencv4`
INCLUDE = `pkg-config --libs protobuf` `pkg-config --libs opencv4`
SRC = ../inference_engine

mnist_mlp.o: mnist_mlp.cpp onnx.o onnx_client.o naive_backend.o image_util.o
	$(CC) $^ $(CFLAGS) $(INCLUDE) $(LIBRARY) -o $@

image_util.o: $(SRC)/image_util.cpp
	$(CC) $^ $(CFLAGS) -o $@ -c

naive_backend.o: $(SRC)/naive_backend.cpp
	$(CC) $^ $(CFLAGS) -o $@ -c

onnx.o: $(SRC)/onnx.cpp	onnx_client.o
	$(CC) $^ $(CFLAGS) -o $@ -c

onnx_client.o: ../external/onnx/onnx/onnx.pb.cc
	$(CC) $^ $(CFLAGS) -o $@ -c

../external/onnx/onnx/onnx.pb.cc: ../external/onnx/onnx/onnx.proto
	protoc -I=../external/onnx/onnx --cpp_out=../external/onnx/onnx ../external/onnx/onnx/onnx.proto

.PHONY :format
format:
	clang-format -i mnist_mlp.cpp

.PHONY :clean
clean:
	$(RM) -rf ./*.o

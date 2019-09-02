SRC = ../inference_engine
CC = clang++
CFLAGS = -std=c++14 -g3 -O2 -Wall -Wextra `pkg-config --cflags protobuf` `pkg-config --cflags opencv4`
INCLUDE = `pkg-config --libs protobuf` `pkg-config --libs opencv4`
SRCS=$(wildcard *.cpp)
HEADERS=$(wildcard *.hpp)

.PHONY: main
main:
	@make clean
	@make test_naive_backend.o
	@make test_image_util.o
	./test_naive_backend.o
	./test_image_util.o

test_image_util.o:	test_image_util.cpp util.o image_util.o
	$(CC) $^ $(CFLAGS) $(INCLUDE) -o $@

test_naive_backend.o: test_naive_backend.cpp util.o naive_backend.o 
	$(CC) $^ $(CFLAGS) -o $@

util.o: util.cpp
	$(CC) $^ $(CFLAGS) -o $@ -c

image_util.o: $(SRC)/image_util.cpp
	$(CC) $^ $(CFLAGS) -o $@ -c

naive_backend.o: $(SRC)/naive_backend.cpp
	$(CC) $^ $(CFLAGS) -o $@ -c

.PHONY :clean
clean:
	$(RM) -rf ./*.o

.PHONY :format
format:
	clang-format $(SRCS) $(HEADERS) -i

# about this repository
- This is a simple [ONNX](https://github.com/onnx/onnx) model inference runtime.

# How to build sample scripts

## Requirements
- cmake (>= 3.15)
- OpenCV (>= 4.0.0)
- protobuf (>= 3.9.1)
- An ONNX file for Multilayer Perceptron model

## Build steps

### MNIST + 3 MLP image classification model sample

Note that I have tested this script with the ONNX model trained with [Chainer mnist example](https://github.com/chainer/chainer/blob/399d861f84fbd32a807ac577fa24170e7fbec8a3/examples/mnist/train_mnist.py) and exported with [onnx-chainer](https://github.com/chainer/onnx-chainer).

```sh
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./example/mnist_mlp.o -i /path/to/mnist/image -m /path/to/onnx_model
```

### ImageNet + VGG19 image classification model sample

Note that I have tested this script with the ONNX from [onnx/models](https://github.com/onnx/models/tree/master/vision/classification/vgg).

```sh
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./example/mnist_mlp.o -i /path/to/image_net/image -m /path/to/onnx_model
```

# How to test

```sh
cd inference_engine/test
git submodule update
make
```

# Supported Operators
- [x] Gemm
- [x] Relu
- [x] Convolution
- [x] MaxPooling
- [x] Dropout
- [x] Softmax
- [x] Reshape (provisional support)

# License
MIT

# about this repository
- This is a simple [ONNX](https://github.com/onnx/onnx) model inference runtime.
- As of September 2019, supported operators are only Gemm and Relu, but I plan to implement more operators.

# How to build sample scripts

## Requirements
- clang++ (>= 8.0.0)
- OpenCV (>= 4.0.0)
- protobuf (>= 3.9.1)
- An ONNX file for Multilayer Perceptron model
  - For development,

## Build steps

### MNIST + 3 MLP image classification model sample

Note that I have tested this script with the ONNX model trained with [Chainer mnist example](https://github.com/chainer/chainer/blob/399d861f84fbd32a807ac577fa24170e7fbec8a3/examples/mnist/train_mnist.py) and exported with [onnx-chainer](https://github.com/chainer/onnx-chainer).

```sh
cd inference_engine/example
git submodule update
make mnist_mlp.o
./mnist_mlp.o -i /path/to/mnist/image -m /path/to/onnx_model
```

### ImageNet + VGG19 image classification model sample

Note that I have tested this script with the ONNX from [onnx/models](https://github.com/onnx/models/tree/master/vision/classification/vgg).

```sh
cd inference_engine/example
git submodule update
make imagenet_vgg19.o
imagenet_vgg19.o -i /path/to/image_net/image -m /path/to/onnx_model
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

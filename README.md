# About this repository
- This is a simple [ONNX](https://github.com/onnx/onnx) model inference runtime.
- This is a just toy project for my study.

# How to build sample scripts

## Requirements
- cmake (>= 3.15)
- OpenCV (>= 4.0.0)
- protobuf (>= 3.9.1)
- An ONNX file for Multilayer Perceptron model

## Build steps

### MNIST + 3 MLP image classification model sample

Note that I have tested this script with the ONNX model trained with [Chainer mnist example](https://github.com/chainer/chainer/blob/399d861f84fbd32a807ac577fa24170e7fbec8a3/examples/mnist/train_mnist.py) and exported with [onnx-chainer](https://github.com/chainer/onnx-chainer).

**Regarding other onnx models, all of them are not gurunteed working well.**

```sh
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
# {Download MNIST data from http://image-net.org/synset}
# {Resize the images to 224 x 224}
./example/mnist_mlp.o -i /path/to/mnist/image -m /path/to/onnx_model
```

### ImageNet + VGG19 image classification model sample

Note that I have tested this script only with the following models with [ImageNet](http://www.image-net.org/) image resized with 224 x 224.

**Regarding other onnx models, all of them are not gurunteed working well.**

- [onnx/models/vgg/vgg19/release 1.1](https://s3.amazonaws.com/download.onnx/models/opset_3/vgg19.tar.gz)
- [onnx/models/vgg/vgg19/release 1.1.2](https://s3.amazonaws.com/download.onnx/models/opset_6/vgg19.tar.gz)
- [onnx/models/vgg/vgg19/release 1.2](https://s3.amazonaws.com/download.onnx/models/opset_7/vgg19.tar.gz)
- [onnx/models/vgg/vgg19/release 1.3](https://s3.amazonaws.com/download.onnx/models/opset_8/vgg19.tar.gz)

```sh
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
# {Download images from http://yann.lecun.com/exdb/mnist/}
# {Convert the binary to jpg image}
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

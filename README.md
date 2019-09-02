# about this repository
- This is a simple [ONNX](https://github.com/onnx/onnx) model inference runtime.
- As of September 2019, supported operators are only Gemm and Relu, but I plan to implement more operators.

# How to build MNIST + MLP example

## Requirements
- clang++ (>= 8.0.0)
- OpenCV (>= 4.0.0)
- protobuf (>= 3.9.1)
- An ONNX file for Multilayer Perceptron model
  - For development, I used the ONNX model trained with [Chainer mnist example](https://github.com/chainer/chainer/blob/399d861f84fbd32a807ac577fa24170e7fbec8a3/examples/mnist/train_mnist.py) and exported with [onnx-chainer](https://github.com/chainer/onnx-chainer).

## Build steps

```sh
cd inference_engine/example
git submodule update
make 
./mnist_mlp.o -i /path/to/mnist/image -m /path/to/onnx_model
```

# Supported Operators
- [x] Gemm
- [x] Relu
- [ ] Convolution
- [ ] MaxPooling

# License
MIT

/*
 * This is an example for ImageNet 1-k + VGG19 model.
 */

#include <algorithm>
#include <numeric>
#include <utility>

#include "../external/cmdline.h"
#include <onnx/onnx_pb.h>

#include "../inference_engine/backend.hpp"
#include "../inference_engine/image_util.hpp"
#include "../inference_engine/inferer.hpp"
#include "../inference_engine/onnx.hpp"

std::shared_ptr<void> allocate_float_array(long long size) {
  std::unique_ptr<float[]> u = std::make_unique<float[]>(size);
  return std::shared_ptr<void>(u.release(), u.get_deleter());
}

int main(int argc, char **argv) {
  cmdline::parser a;
  a.add<std::string>("image_path", 'i',
                     "The file path of the image which you want to infer",
                     true);
  a.add<std::string>("model_path", 'm',
                     "The file path of the ONNX model which you want to use to "
                     "infer the image",
                     true);
  a.parse_check(argc, argv);

  const std::string image_path = a.get<std::string>("image_path");
  const std::string model_path = a.get<std::string>("model_path");

  cv::Mat image_mat = cv::imread(image_path, cv::IMREAD_COLOR);
  if (!image_mat.data) {
    std::cout << "Invalid image: " << image_path << std::endl;
    return -1;
  }

  ::onnx::ModelProto model;
  try {
    model = inference_engine::onnx::load_onnx_model_from_file(model_path);
  } catch (std::runtime_error e) {
    std::cout << "ONNX LOAD ERROR: " << e.what() << std::endl;
    return -1;
  }

  ::onnx::GraphProto graph = model.graph();
  std::vector<inference_engine::onnx::node> nodes =
      inference_engine::onnx::abstract_all_nodes(graph);
  std::map<std::string, inference_engine::onnx::parameter> table;
  inference_engine::onnx::abstract_parameter_table(graph, table);

  try {
    inference_engine::onnx::initialize_parameter_table(graph, table);
  } catch (std::out_of_range e) {
    std::cout << "out_of_range at initialize_parameter_table: " << e.what()
              << std::endl;
    return -1;
  }

  // input preprocessing
  const int channel_num = 3;
  const int height = 224;
  const int width = 224;

  cv::resize(image_mat, image_mat, cv::Size(width, height));
  image_mat.convertTo(image_mat, CV_32FC3);
  std::shared_ptr<void> data =
      allocate_float_array(channel_num * height * width);
  float *image = static_cast<float *>(data.get());
  image_mat -= cv::Scalar(103.939, 116.779, 123.68); // subtract BGR mean
  inference_engine::image_util::rgb_image_to_chw(image_mat, image);
  table.at(nodes[0].input[0]).data = image;

  long stride, pad, kernel, m, n, k, x_h, x_w, c_in, c_out;
  for (inference_engine::onnx::node const &node : nodes) {
    if (node.op_type == inference_engine::onnx::OP_TYPE::Conv) {
      c_in = table.at(node.input[0]).dims[1];
      x_h = table.at(node.input[0]).dims[2];
      x_w = table.at(node.input[0]).dims[3];
      c_out = table.at(node.input[1]).dims[0];
      stride = static_cast<long *>(node.attributes.at("strides").data)[0];
      pad = static_cast<long *>(node.attributes.at("pads").data)[0];
      kernel = static_cast<long *>(node.attributes.at("kernel_shape").data)[0];
      assert(c_in == table.at(node.input[1]).dims[1]);

      std::pair<long, long> y_dims =
          inference_engine::inferer::calculate_conv_matrix_dims(
              x_h, x_w, kernel, pad, stride);

      if (table.find(node.output[0]) == table.end()) {
        inference_engine::onnx::add_new_parameter(
            node.output[0], {c_out, y_dims.first, y_dims.second},
            table.at(node.input[0]).data_type, table);
      } else {
        inference_engine::onnx::reset_parameter_data(node.output[0], table);
      }

      inference_engine::backend::conv(
          c_in, c_out, x_h, x_w, y_dims.first, y_dims.second, kernel, pad,
          stride,
          static_cast<float *>(table.at(node.input[0]).data), // x
          static_cast<float *>(table.at(node.input[1]).data), // w
          static_cast<float *>(table.at(node.input[2]).data), // b
          static_cast<float *>(table.at(node.output[0]).data) // y
      );
    } else if (node.op_type == inference_engine::onnx::OP_TYPE::Gemm) {
      m = table.at(node.input[1]).dims[0];
      k = table.at(node.input[1]).dims[1];
      n = table.at(node.input[0]).dims[0];

      if (table.find(node.output[0]) == table.end()) {
        inference_engine::onnx::add_new_parameter(
            node.output[0], {n, m}, table.at(node.input[0]).data_type, table);
      } else {
        inference_engine::onnx::reset_parameter_data(node.output[0], table);
      }

      inference_engine::backend::gemm(
          m, n, k,
          static_cast<float *>(table.at(node.input[1]).data),  // A
          static_cast<float *>(table.at(node.input[0]).data),  // B
          static_cast<float *>(table.at(node.output[0]).data), // C
          static_cast<float *>(table.at(node.input[2]).data)   // D
      );

    } else if (node.op_type == inference_engine::onnx::OP_TYPE::Relu) {
      // TODO Refactor
      if (table.at(node.input[0]).dims.size() == 3) {
        c_out = table.at(node.input[0]).dims[0];
        x_h = table.at(node.input[0]).dims[1];
        x_w = table.at(node.input[0]).dims[2];

        if (table.find(node.output[0]) == table.end()) {
          inference_engine::onnx::add_new_parameter(
              node.output[0], {1, c_out, x_h, x_w},
              table.at(node.input[0]).data_type, table);
        } else {
          inference_engine::onnx::reset_parameter_data(node.output[0], table);
        }

        inference_engine::backend::relu(
            c_out * x_h * x_w,
            static_cast<float *>(table.at(node.input[0]).data),
            static_cast<float *>(table.at(node.output[0]).data));
      } else {
        x_h = table.at(node.input[0]).dims[0];
        x_w = table.at(node.input[0]).dims[1];

        if (table.find(node.output[0]) == table.end()) {
          inference_engine::onnx::add_new_parameter(
              node.output[0], {x_h, x_w}, table.at(node.input[0]).data_type,
              table);
        } else {
          inference_engine::onnx::reset_parameter_data(node.output[0], table);
        }

        inference_engine::backend::relu(
            x_h * x_w, static_cast<float *>(table.at(node.input[0]).data),
            static_cast<float *>(table.at(node.output[0]).data));
      }
    } else if (node.op_type == inference_engine::onnx::OP_TYPE::MaxPool) {
      c_out = table.at(node.input[0]).dims[1];
      x_h = table.at(node.input[0]).dims[2];
      x_w = table.at(node.input[0]).dims[3];
      stride = static_cast<long *>(node.attributes.at("strides").data)[0];
      pad = static_cast<long *>(node.attributes.at("pads").data)[0];
      kernel = static_cast<long *>(node.attributes.at("kernel_shape").data)[0];

      std::pair<long, long> y_dims =
          inference_engine::inferer::calculate_conv_matrix_dims(
              x_h, x_w, kernel, pad, stride);

      if (table.find(node.output[0]) == table.end()) {
        inference_engine::onnx::add_new_parameter(
            node.output[0], {1, c_out, y_dims.first, y_dims.second},
            table.at(node.input[0]).data_type, table);
      } else {
        inference_engine::onnx::reset_parameter_data(node.output[0], table);
      }

      inference_engine::backend::max_pool(
          c_out, x_h, x_w, y_dims.first, y_dims.second, kernel, pad, stride,
          static_cast<float *>(table.at(node.input[0]).data), // x
          static_cast<float *>(table.at(node.output[0]).data) // y
      );
    } else if (node.op_type == inference_engine::onnx::OP_TYPE::Reshape) {
      // TODO Support other than 2 dimension matrix
      if (table.find(node.output[0]) == table.end()) {
        inference_engine::onnx::add_new_parameter(
            node.output[0],
            {static_cast<long *>(table.at(node.input[1]).data)[0],
             static_cast<long *>(table.at(node.input[1]).data)[1]},
            table.at(node.input[0]).data_type, table);
      } else {
        inference_engine::onnx::reset_parameter_data(node.output[0], table);
      }

      table.at(node.output[0]).data = table.at(node.input[0]).data;
    } else if (node.op_type == inference_engine::onnx::OP_TYPE::Dropout) {
      // TODO Support other than 2 dimension matrix
      x_h = table.at(node.input[0]).dims[0];
      x_w = table.at(node.input[0]).dims[1];
      float ratio = static_cast<float *>(node.attributes.at("ratio").data)[0];

      // output
      if (table.find(node.output[0]) == table.end()) {
        inference_engine::onnx::add_new_parameter(
            node.output[0], {x_h, x_w}, table.at(node.input[0]).data_type,
            table);
      } else {
        inference_engine::onnx::reset_parameter_data(node.output[0], table);
      }

      // mask
      if (table.find(node.output[1]) == table.end()) {
        inference_engine::onnx::add_new_parameter(
            node.output[1], {x_h, x_w}, table.at(node.input[0]).data_type,
            table);
      } else {
        inference_engine::onnx::reset_parameter_data(node.output[1], table);
      }

      inference_engine::backend::drop_out(
          x_h * x_w, ratio,
          static_cast<float *>(table.at(node.input[0]).data),  // x
          static_cast<float *>(table.at(node.output[0]).data), // y
          static_cast<float *>(table.at(node.output[1]).data)  // mask
      );
    } else if (node.op_type == inference_engine::onnx::OP_TYPE::Softmax) {
      x_h = table.at(node.input[0]).dims[0];
      x_w = table.at(node.input[0]).dims[1];

      if (table.find(node.output[0]) == table.end()) {
        inference_engine::onnx::add_new_parameter(
            node.output[0], {x_h, x_w}, table.at(node.input[0]).data_type,
            table);
      } else {
        inference_engine::onnx::reset_parameter_data(node.output[0], table);
      }

      inference_engine::backend::softmax(
          x_h * x_w,
          static_cast<float *>(table.at(node.input[0]).data), // x
          static_cast<float *>(table.at(node.output[0]).data) // y
      );
    } else {
      std::cout << "not supported operator: " << node.op_type << std::endl;
      return -1;
    }
  }

  std::cout << "inference result" << std::endl;
  float *result = static_cast<float *>(table.at(graph.output()[0].name()).data);
  std::multimap<float, int, std::greater<float>> sorted_map;
  for (int i = 0; i < 1000; ++i) {
    sorted_map.insert(std::make_pair(result[i], i));
  }

  int i = 0;
  for (auto const &element : sorted_map) {
    if (i == 10)
      break;
    std::cout << element.second << "   :   " << element.first << std::endl;
    ++i;
  }

  return 0;
}
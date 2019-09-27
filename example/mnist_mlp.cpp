/*
 * This is an example for MNIST + 3 MLP (Gemm + Relu) model.
 */

#include <numeric>

#include "../external/cmdline.h"
#include "../external/onnx/onnx/onnx.pb.h"

#include "../inference_engine/backend.hpp"
#include "../inference_engine/image_util.hpp"
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

  // convert to 32bit
  image_mat.convertTo(image_mat, CV_32FC3);
  std::shared_ptr<void> data =
      allocate_float_array(image_mat.rows * image_mat.cols);
  float *image = static_cast<float *>(data.get());
  inference_engine::image_util::gray_image_to_hw(image_mat, image);
  table.at(nodes[0].input[0]).data = image;

  int m, n, k;
  for (inference_engine::onnx::node const &node : nodes) {
    if (node.op_type == inference_engine::onnx::OP_TYPE::Gemm) {
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
      m = table.at(node.input[0]).dims[0];
      n = table.at(node.input[0]).dims[1];

      if (table.find(node.output[0]) == table.end()) {
        inference_engine::onnx::add_new_parameter(
            node.output[0], {m, n}, table.at(node.input[0]).data_type, table);
      } else {
        inference_engine::onnx::reset_parameter_data(node.output[0], table);
      }

      inference_engine::backend::relu(
          m * n, static_cast<float *>(table.at(node.input[0]).data),
          static_cast<float *>(table.at(node.output[0]).data));
    } else {
      std::cout << "not supported operator: " << node.op_type << std::endl;
      return -1;
    }
  }

  for (int i = 0; i < 10; ++i) {
    std::cout << static_cast<float *>(
                     table.at(graph.output()[0].name()).data)[i]
              << std::endl;
  }

  return 0;
}
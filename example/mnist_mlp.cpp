/*
 * This is an example for MNIST + 3 MLP (Gemm + Relu) model.
*/

#include <numeric>

#include "../external/cmdline.h"
#include "../external/onnx/onnx/onnx.pb.h"

#include "../inference_engine/backend.hpp"
#include "../inference_engine/image_util.hpp"
#include "../inference_engine/onnx.hpp"

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

  // convert to 32bit
  image_mat.convertTo(image_mat, CV_32FC3);
  // FIXME workaround
  float *image = new float[image_mat.rows * image_mat.cols];
  memset(image, 0, sizeof(float) * image_mat.rows * image_mat.cols);
  inference_engine::image_util::gray_image_to_hw(image_mat, image);

  ::onnx::ModelProto model;
  try {
    model = inference_engine::onnx::load_onnx_model_from_file(model_path);
  } catch (std::runtime_error e) {
    std::cout << "ONNX LOAD ERROR: " << e.what() << std::endl;
    return -1;
  }

  auto graph = model.graph();
  std::vector<inference_engine::onnx::node> nodes =
      inference_engine::onnx::abstract_all_nodes_from_onnx_model(graph);
  std::map<std::string, inference_engine::onnx::parameter> table;
  inference_engine::onnx::abstract_parameter_table_from_onnx_model(graph,
                                                                   table);

  std::string first_input_name = graph.input()[graph.input_size() - 1].name();
  int first_input_hight = graph.input()[graph.input_size() - 1]
                              .type()
                              .tensor_type()
                              .shape()
                              .dim(1)
                              .dim_value();
  int first_input_width = graph.input()[graph.input_size() - 1]
                              .type()
                              .tensor_type()
                              .shape()
                              .dim(0)
                              .dim_value();

  inference_engine::onnx::parameter input_parameter =
      inference_engine::onnx::parameter(
          first_input_name, {first_input_hight, first_input_width},
          ::onnx::TensorProto_DataType::TensorProto_DataType_FLOAT, image);
  table.insert(std::make_pair(first_input_name, input_parameter));

  std::string input_name;
  std::string output_parameter_name;
  float *output_data;
  int m, n, k;

  for (int i = 0; i < (int)nodes.size(); ++i) {
    input_name = nodes[i].name;
    if (nodes[i].op_type == inference_engine::onnx::OP_TYPE::Gemm) {
      m = table.at(nodes[i].input[1]).dims[0];
      k = table.at(nodes[i].input[1]).dims[1];
      n = table.at(nodes[i].input[0]).dims[1];

      output_parameter_name = nodes[i].output[0];
      output_data = new float[m * n];
      memset(output_data, 0, sizeof(float) * m * n);
      inference_engine::onnx::parameter output_parameter =
          inference_engine::onnx::parameter(
              output_parameter_name, {m, n},
              ::onnx::TensorProto_DataType::TensorProto_DataType_FLOAT,
              output_data);
      table.insert(std::make_pair(output_parameter_name, output_parameter));
      inference_engine::backend::gemm(m, n, k,
                                      table.at(nodes[i].input[1]).data,  // A
                                      table.at(nodes[i].input[0]).data,  // B
                                      table.at(nodes[i].output[0]).data, // C
                                      table.at(nodes[i].input[2]).data   // D
      );
    } else if (nodes[i].op_type == inference_engine::onnx::OP_TYPE::Relu) {
      m = table.at(nodes[i].input[0]).dims[0];
      n = table.at(nodes[i].input[0]).dims[1];

      output_parameter_name = nodes[i].output[0];
      // FIXME workaround
      output_data = new float[m * n];
      memset(output_data, 0, sizeof(float) * m * n);
      inference_engine::onnx::parameter output_parameter =
          inference_engine::onnx::parameter(
              output_parameter_name, {m, n},
              ::onnx::TensorProto_DataType::TensorProto_DataType_FLOAT,
              output_data);
      table.insert(
          std::make_pair(output_parameter_name, std::move(output_parameter)));

      inference_engine::backend::relu(m, n, table.at(nodes[i].input[0]).data,
                                      table.at(nodes[i].output[0]).data);
    } else {
      std::cout << "not supported operator: " << nodes[i].op_type << std::endl;
      return -1;
    }
  }

  for (int i = 0; i < 10; ++i) {
    std::cout << table.at(graph.output()[0].name()).data[i] << std::endl;
  }

  return 0;
}
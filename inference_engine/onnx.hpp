#ifndef ONNX_HPP
#define ONNX_HPP

#include "../external/onnx/onnx/onnx.pb.h"
#include <vector>

namespace inference_engine {
namespace onnx {

enum OP_TYPE { Gemm, Relu };

struct parameter {
  std::string name;
  std::vector<int> dims;
  ::google::protobuf::int32 data_type;
  void *data;
  long long total_size;

  parameter(std::string name, std::vector<int> dims,
            ::google::protobuf::int32 data_type, void *data,
            long long total_size)
      : name(name), dims(dims), data_type(data_type), data(data),
        total_size(total_size) {}
};

struct node {
  std::string name;
  inference_engine::onnx::OP_TYPE op_type;
  ::google::protobuf::RepeatedPtrField<::std::string> input;
  ::google::protobuf::RepeatedPtrField<::std::string> output;

  node(std::string name, inference_engine::onnx::OP_TYPE op_type,
       ::google::protobuf::RepeatedPtrField<::std::string> input,
       ::google::protobuf::RepeatedPtrField<::std::string> output)
      : name(name), op_type(op_type), input(input), output(output) {}
};

::onnx::ModelProto load_onnx_model_from_file(std::string const &model_path);

void unpack_data_from_raw_data(const ::onnx::TensorProto &tensor,
                               ::google::protobuf::int32 total_size,
                               void *output);

void abstract_parameter_table_from_onnx_model(
    ::onnx::GraphProto &graph,
    std::map<std::string, inference_engine::onnx::parameter> &table);

void abstract_parameters(
    ::google::protobuf::RepeatedPtrField<::onnx::ValueInfoProto> it,
    std::map<std::string, inference_engine::onnx::parameter> &table);

void add_new_parameter(
    std::string parameter_name, std::vector<int> dims,
    ::google::protobuf::int32 data_type,
    std::map<std::string, inference_engine::onnx::parameter> &table);

void initialize_parameter_table_from_onnx_model(
    ::onnx::GraphProto &graph,
    std::map<std::string, inference_engine::onnx::parameter> &table);

inference_engine::onnx::OP_TYPE convert_op_type(std::string op_type);

std::vector<inference_engine::onnx::node>
abstract_all_nodes_from_onnx_model(::onnx::GraphProto &graph);
} // namespace onnx
} // namespace inference_engine
#endif

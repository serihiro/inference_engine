#ifndef ONNX_HPP
#define ONNX_HPP

#include <map>
#include <onnx/onnx_pb.h>
#include <vector>

namespace inference_engine {
namespace onnx {

enum OP_TYPE { Gemm, Relu, Conv, MaxPool, Reshape, Dropout, Softmax };

const std::map<::google::protobuf::string, inference_engine::onnx::OP_TYPE>
    OP_TYPE_MAP = {{"Gemm", inference_engine::onnx::OP_TYPE::Gemm},
                   {"Relu", inference_engine::onnx::OP_TYPE::Relu},
                   {"Conv", inference_engine::onnx::OP_TYPE::Conv},
                   {"MaxPool", inference_engine::onnx::OP_TYPE::MaxPool},
                   {"Reshape", inference_engine::onnx::OP_TYPE::Reshape},
                   {"Dropout", inference_engine::onnx::OP_TYPE::Dropout},
                   {"Softmax", inference_engine::onnx::OP_TYPE::Softmax}};

struct parameter {
  std::string name;
  std::vector<long> dims;
  ::google::protobuf::int32 data_type;
  void *data;
  long long total_size;

  parameter(std::string name, std::vector<long> dims,
            ::google::protobuf::int32 data_type, void *data,
            long long total_size)
      : name(name), dims(dims), data_type(data_type), data(data),
        total_size(total_size) {}
};

struct attribute {
  std::string name;
  ::onnx::AttributeProto_AttributeType data_type;
  void *data;

  attribute(std::string name, ::onnx::AttributeProto_AttributeType data_type,
            void *data)
      : name(name), data_type(data_type), data(data) {}
};

struct node {
  std::string name;
  inference_engine::onnx::OP_TYPE op_type;
  ::google::protobuf::RepeatedPtrField<::std::string> input;
  ::google::protobuf::RepeatedPtrField<::std::string> output;
  std::map<std::string, attribute> attributes;

  node(std::string name, inference_engine::onnx::OP_TYPE op_type,
       ::google::protobuf::RepeatedPtrField<::std::string> input,
       ::google::protobuf::RepeatedPtrField<::std::string> output)
      : name(name), op_type(op_type), input(input), output(output) {}

  node(std::string name, inference_engine::onnx::OP_TYPE op_type,
       ::google::protobuf::RepeatedPtrField<::std::string> input,
       ::google::protobuf::RepeatedPtrField<::std::string> output,
       std::map<std::string, attribute> attributes)
      : name(name), op_type(op_type), input(input), output(output),
        attributes(attributes) {}
};

::onnx::ModelProto load_onnx_model_from_file(std::string const &model_path);

void unpack_data_from_raw_data(const ::onnx::TensorProto &tensor,
                               ::google::protobuf::int32 total_size,
                               void *output, int data_unit_size);

void abstract_parameter_table(
    ::onnx::GraphProto &graph,
    std::map<std::string, inference_engine::onnx::parameter> &table);

void abstract_parameters(
    ::google::protobuf::RepeatedPtrField<::onnx::ValueInfoProto> it,
    std::map<std::string, inference_engine::onnx::parameter> &table);

void add_new_parameter(
    std::string parameter_name, std::vector<long> dims,
    ::google::protobuf::int32 data_type,
    std::map<std::string, inference_engine::onnx::parameter> &table);

void reset_parameter_data(
    std::string target_parameter_name,
    std::map<std::string, inference_engine::onnx::parameter> &table);

void initialize_parameter_table(
    ::onnx::GraphProto &graph,
    std::map<std::string, inference_engine::onnx::parameter> &table);

inference_engine::onnx::OP_TYPE convert_op_type(std::string op_type);

std::vector<inference_engine::onnx::node>
abstract_all_nodes(::onnx::GraphProto &graph);

std::map<std::string, inference_engine::onnx::attribute>
abstract_attributes(::onnx::NodeProto node);
} // namespace onnx
} // namespace inference_engine
#endif

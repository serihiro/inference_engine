#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

#include "onnx.hpp"

namespace inference_engine {
namespace onnx {
::onnx::ModelProto load_onnx_model_from_file(std::string const &model_path) {
  std::ifstream ifs(model_path, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Invalid file name: " + model_path);
  }
  ::google::protobuf::io::IstreamInputStream iis(&ifs);
  ::google::protobuf::io::CodedInputStream cis(&iis);
  cis.SetTotalBytesLimit(std::numeric_limits<int>::max(),
                         std::numeric_limits<int>::max());

  ::onnx::ModelProto onnx_model;
  if (!onnx_model.ParseFromCodedStream(&cis)) {
    throw std::runtime_error("ONNX file parse error: " + model_path);
  }
  return onnx_model;
}

void unpack_data_from_raw_data(const ::onnx::TensorProto &tensor,
                               ::google::protobuf::int32 total_size,
                               void *output) {
  assert(tensor.raw_data().length() ==
         static_cast<decltype(tensor.raw_data().length())>(total_size * 4));

  std::copy(tensor.raw_data().begin(), tensor.raw_data().end(),
            static_cast<char *>(output));
}

void abstract_parameter_table(
    ::onnx::GraphProto &graph,
    std::map<std::string, inference_engine::onnx::parameter> &table) {

  abstract_parameters(graph.input(), table);
  abstract_parameters(graph.output(), table);
} // namespace onnx

void abstract_parameters(
    ::google::protobuf::RepeatedPtrField<::onnx::ValueInfoProto> it,
    std::map<std::string, inference_engine::onnx::parameter> &table) {

  std::vector<int> dims;
  for (::onnx::ValueInfoProto const &value_info : it) {
    dims.clear();
    for (::onnx::TensorShapeProto_Dimension const &dim :
         value_info.type().tensor_type().shape().dim()) {
      dims.push_back(dim.dim_value());
    }
    add_new_parameter(value_info.name(), dims,
                      value_info.type().tensor_type().elem_type(), table);
  }
}

void add_new_parameter(
    std::string parameter_name, std::vector<int> dims,
    ::google::protobuf::int32 data_type,
    std::map<std::string, inference_engine::onnx::parameter> &table) {

  void *data;
  long long total_size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  if (data_type == ::onnx::TensorProto_DataType::TensorProto_DataType_FLOAT) {
    data = static_cast<void *>(new float[total_size]);
    memset(data, 0, sizeof(float) * total_size);
  } else if (data_type ==
                 ::onnx::TensorProto_DataType::TensorProto_DataType_INT8 ||
             data_type ==
                 ::onnx::TensorProto_DataType::TensorProto_DataType_INT16 ||
             data_type ==
                 ::onnx::TensorProto_DataType::TensorProto_DataType_INT32) {
    data = static_cast<void *>(new int[total_size]);
    memset(data, 0, sizeof(int) * total_size);
  } else if (data_type ==
             ::onnx::TensorProto_DataType::TensorProto_DataType_INT64) {
    data = static_cast<void *>(new long[total_size]);
    memset(data, 0, sizeof(long) * total_size);
  } else {
    throw std::runtime_error("un supported type: " + std::to_string(data_type));
  }

  table.insert(std::make_pair(
      parameter_name, inference_engine::onnx::parameter(
                          parameter_name, dims, data_type, data, total_size)));
}

void reset_parameter_data(
    std::string target_parameter_name,
    std::map<std::string, inference_engine::onnx::parameter> &table) {
  if (table.at(target_parameter_name).data_type ==
      ::onnx::TensorProto_DataType::TensorProto_DataType_FLOAT) {
    memset(table.at(target_parameter_name).data, 0,
           sizeof(float) * table.at(target_parameter_name).total_size);
  } else if (table.at(target_parameter_name).data_type ==
                 ::onnx::TensorProto_DataType::TensorProto_DataType_INT8 ||
             table.at(target_parameter_name).data_type ==
                 ::onnx::TensorProto_DataType::TensorProto_DataType_INT16 ||
             table.at(target_parameter_name).data_type ==
                 ::onnx::TensorProto_DataType::TensorProto_DataType_INT32) {
    memset(table.at(target_parameter_name).data, 0,
           sizeof(int) * table.at(target_parameter_name).total_size);
  } else if (table.at(target_parameter_name).data_type ==
             ::onnx::TensorProto_DataType::TensorProto_DataType_INT64) {
    memset(table.at(target_parameter_name).data, 0,
           sizeof(long) * table.at(target_parameter_name).total_size);
  } else {
    throw std::runtime_error(
        "un supported type: " +
        std::to_string(table.at(target_parameter_name).data_type));
  }
}

void initialize_parameter_table(
    ::onnx::GraphProto &graph,
    std::map<std::string, inference_engine::onnx::parameter> &table) {
  for (::onnx::TensorProto const &tensor : graph.initializer()) {
    unpack_data_from_raw_data(tensor, table.at(tensor.name()).total_size,
                              table.at(tensor.name()).data);
  }
}

inference_engine::onnx::OP_TYPE convert_op_type(std::string op_type) {
  if (op_type == "Gemm") {
    return inference_engine::onnx::OP_TYPE::Gemm;
  } else if (op_type == "Relu") {
    return inference_engine::onnx::OP_TYPE::Relu;
  } else {
    throw std::runtime_error("node supported op_type: " + op_type);
  }
}

std::vector<inference_engine::onnx::node>
abstract_all_nodes(::onnx::GraphProto &graph) {
  std::vector<inference_engine::onnx::node> nodes;

  for (auto const &node : graph.node()) {
    nodes.push_back(inference_engine::onnx::node(
        node.name(), convert_op_type(node.op_type()), node.input(),
        node.output()));
  }

  return nodes;
}
} // namespace onnx
} // namespace inference_engine
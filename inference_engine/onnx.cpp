#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <map>
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

void abstract_parameter_table_from_onnx_model(
    ::onnx::GraphProto &graph,
    std::map<std::string, inference_engine::onnx::parameter> &table) {
  void *data;
  long long total_size;
  for (::onnx::TensorProto const &tensor : graph.initializer()) {
    std::vector<int> dims(tensor.dims().begin(), tensor.dims().end());
    total_size =
        std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    ::google::protobuf::int32 data_type = tensor.data_type();

    // FIXME workaround
    data = static_cast<void *>(new float[total_size]);
    memset(data, 0, sizeof(float) * total_size);
  
    // TODO I want to abstract typename
    if (data_type == ::onnx::TensorProto_DataType::TensorProto_DataType_FLOAT) {
      unpack_data_from_raw_data(tensor, total_size, data);
    } else {
      throw std::runtime_error("un supported type: " +
                               std::to_string(data_type));
    }

    table.insert(std::make_pair(
        tensor.name(),
        inference_engine::onnx::parameter(tensor.name(), dims, data_type,
                                          static_cast<float *>(data))));
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
abstract_all_nodes_from_onnx_model(::onnx::GraphProto &graph) {
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
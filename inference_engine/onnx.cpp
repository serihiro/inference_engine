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
                               void *output, int data_unit_size) {
  assert(tensor.raw_data().length() ==
         static_cast<decltype(tensor.raw_data().length())>(total_size *
                                                           data_unit_size));

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

  for (::onnx::ValueInfoProto const &value_info : it) {
    std::vector<long> dims(
        value_info.type().tensor_type().shape().dim().size());

    std::transform(value_info.type().tensor_type().shape().dim().begin(),
                   value_info.type().tensor_type().shape().dim().end(),
                   dims.begin(),
                   ([](::onnx::TensorShapeProto_Dimension dim) -> long {
                     return dim.dim_value();
                   }));

    add_new_parameter(value_info.name(), dims,
                      value_info.type().tensor_type().elem_type(), table);
  }
}

void add_new_parameter(
    std::string parameter_name, std::vector<long> dims,
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
    if (tensor.has_raw_data()) {
      if (tensor.data_type() ==
          ::onnx::TensorProto_DataType::TensorProto_DataType_FLOAT) {
        unpack_data_from_raw_data(tensor, table.at(tensor.name()).total_size,
                                  table.at(tensor.name()).data, sizeof(float));
      } else if (tensor.data_type() ==
                     ::onnx::TensorProto_DataType::TensorProto_DataType_INT8 ||
                 tensor.data_type() ==
                     ::onnx::TensorProto_DataType::TensorProto_DataType_INT16 ||
                 tensor.data_type() ==
                     ::onnx::TensorProto_DataType::TensorProto_DataType_INT32) {
        unpack_data_from_raw_data(tensor, table.at(tensor.name()).total_size,
                                  table.at(tensor.name()).data, sizeof(int));
      } else if (tensor.data_type() ==
                 ::onnx::TensorProto_DataType::TensorProto_DataType_INT64) {
        unpack_data_from_raw_data(tensor, table.at(tensor.name()).total_size,
                                  table.at(tensor.name()).data, sizeof(long));
      } else {
        throw std::runtime_error("un supported type: " +
                                 std::to_string(tensor.data_type()));
      }
    }
  }
}

inference_engine::onnx::OP_TYPE convert_op_type(std::string op_type) {
  if (op_type == "Gemm") {
    return inference_engine::onnx::OP_TYPE::Gemm;
  } else if (op_type == "Relu") {
    return inference_engine::onnx::OP_TYPE::Relu;
  } else if (op_type == "Conv") {
    return inference_engine::onnx::OP_TYPE::Conv;
  } else if (op_type == "MaxPool") {
    return inference_engine::onnx::OP_TYPE::MaxPool;
  } else if (op_type == "Dropout") {
    return inference_engine::onnx::OP_TYPE::Dropout;
  } else if (op_type == "Softmax") {
    return inference_engine::onnx::OP_TYPE::Softmax;
  } else if (op_type == "Reshape") {
    return inference_engine::onnx::OP_TYPE::Reshape;
  } else {
    throw std::runtime_error("node supported op_type: " + op_type);
  }
}

std::vector<inference_engine::onnx::node>
abstract_all_nodes(::onnx::GraphProto &graph) {
  std::vector<inference_engine::onnx::node> nodes;

  for (::onnx::NodeProto const &node : graph.node()) {
    nodes.push_back(inference_engine::onnx::node(
        node.name(), convert_op_type(node.op_type()), node.input(),
        node.output(), abstract_attributes(node)));
  }

  return nodes;
}

std::map<std::string, inference_engine::onnx::attribute>
abstract_attributes(::onnx::NodeProto node) {
  if (node.attribute_size() == 0) {
    return std::map<std::string, inference_engine::onnx::attribute>();
  }

  std::map<std::string, inference_engine::onnx::attribute> result;
  void *data;
  for (::onnx::AttributeProto const &attribute : node.attribute()) {
    if (attribute.type() == ::onnx::AttributeProto_AttributeType::
                                AttributeProto_AttributeType_FLOAT) {
      float *tmp_data = new float;
      *tmp_data = attribute.f();
      data = static_cast<void *>(tmp_data);
    } else if (attribute.type() == ::onnx::AttributeProto_AttributeType::
                                       AttributeProto_AttributeType_INT) {
      long *tmp_data = new long;
      *tmp_data = attribute.i();
      data = static_cast<void *>(tmp_data);
    } else if (attribute.type() == ::onnx::AttributeProto_AttributeType::
                                       AttributeProto_AttributeType_FLOATS) {
      data = static_cast<void *>(new float[attribute.floats_size()]);
      memset(data, 0, sizeof(float) * attribute.floats_size());
      std::copy(attribute.floats().begin(), attribute.floats().end(),
                static_cast<float *>(data));
    } else if (attribute.type() == ::onnx::AttributeProto_AttributeType::
                                       AttributeProto_AttributeType_INTS) {
      data = static_cast<void *>(new long[attribute.ints_size()]);
      memset(data, 0, sizeof(long) * attribute.ints_size());
      std::copy(attribute.ints().begin(), attribute.ints().end(),
                static_cast<long *>(data));
    } else {
      throw std::runtime_error("node supported attribute_type: " +
                               std::to_string(attribute.type()));
    }

    result.insert(std::make_pair(
        attribute.name(), inference_engine::onnx::attribute(
                              attribute.name(), attribute.type(), data)));
  }

  return result;
}
} // namespace onnx
} // namespace inference_engine
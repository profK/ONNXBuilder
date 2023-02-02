#pragma once
#include <fstream>
#include "onnx.pb.h"


using namespace std;

namespace onxb
{
    
    class MLP
    {
    private:
        onnx::ModelProto protobuf;
        int input_layer_size;
        int output_layer_size;
        int hidden_layer_size;
        int num_hidden_layer;
        

        onnx::TensorProto*  add_weights_init(string name,int depth,int width)
        {
            onnx::TensorProto* node = protobuf.mutable_graph()->add_initializer();
            node->mutable_name()->assign(name);
            node->add_dims(depth); //depth
            node->add_dims(width); // width
            for (int i = 0; i < depth * width; i++)
                node->add_float_data(1.0);
            node->set_data_type(
                onnx::TensorProto::DataType::TensorProto_DataType_FLOAT);
            //data will be set when we run the MLP
            return node;
        }

        onnx::TensorProto*  add_bias_init(float bias)
        {
            onnx::TensorProto* node = protobuf.mutable_graph()->add_initializer();
            node->mutable_name()->assign("bias");
            node->add_float_data(bias);
            node->set_data_type(
                onnx::TensorProto::DataType::TensorProto_DataType_FLOAT);
            return node;
        }

        void add_mult_op(string name, string input1, string input2,
            string output)
        {
            onnx::NodeProto* node = protobuf.mutable_graph()->add_node();
            node->mutable_name()->assign(name);
            node->add_input(input1);
            node->add_input(input2);
            node->add_output(output);
            node->mutable_op_type()->assign("MatMul");
        }

        void add_add_op(string  name, string input1, string input2,
            string output)
        {
            onnx::NodeProto* node = protobuf.mutable_graph()->add_node();
            node->mutable_name()->assign(name);
            node->add_input(input1);
            node->add_input(input2);
            node->add_output(output);
            node->mutable_op_type()->assign("Add");
        }

        void add_relu_op(string  name, string input1,
            string output)
        {
            onnx::NodeProto* node = protobuf.mutable_graph()->add_node();
            node->mutable_name()->assign(name);
            node->add_input(input1);
            node->add_output(output);
            node->mutable_op_type()->assign("Relu");
        }

        void add_sigmoid_op(string  name, string input1,
            string output)
        {
            onnx::NodeProto* node = protobuf.mutable_graph()->add_node();
            node->mutable_name()->assign(name);
            node->add_input(input1);
            node->add_output(output);
            node->mutable_op_type()->assign("Sigmoid");
        }

        
        
    public:
       

        MLP(int input_layer,int hidden_layer,int hidden_layer_count,int output_layer,
            float bias):
            input_layer_size(input_layer),output_layer_size(output_layer),hidden_layer_size(hidden_layer),num_hidden_layer(hidden_layer_count)
        {
            protobuf.clear_graph();
            protobuf.mutable_graph()->set_name("MLP Graph");

            onnx::OperatorSetIdProto opset;
            opset.set_domain("");
            opset.set_version(10);
            protobuf.mutable_opset_import()->Add()->CopyFrom(opset);

            protobuf.set_ir_version(5);
            
            onnx::ValueInfoProto* input =protobuf.mutable_graph()->add_input();
            auto input_type =
                new onnx::TypeProto_Tensor();
            input_type->set_elem_type(onnx::TensorProto::DataType::TensorProto_DataType_FLOAT);
            input_type->mutable_shape()->add_dim()->set_dim_value(input_layer_size);
            input->mutable_type()->set_allocated_tensor_type(input_type);
            input->mutable_name()->assign("input");

            onnx::ValueInfoProto* output =protobuf.mutable_graph()->add_output();
            auto output_type =
                new onnx::TypeProto_Tensor();
            output_type->set_elem_type(onnx::TensorProto::DataType::TensorProto_DataType_FLOAT);
            output_type->mutable_shape()->add_dim()->set_dim_value(output_layer_size);
            output->mutable_type()->set_allocated_tensor_type(output_type);
            output->mutable_name()->assign("output");

            add_weights_init("in_weights", input_layer, hidden_layer);
            for(int layer_num=0;layer_num<hidden_layer_count-1;layer_num++)
            {
                string numstr = to_string(layer_num);
                add_weights_init(string("hidden_weights_") + numstr, hidden_layer,
                hidden_layer);
            }
            add_weights_init("out_weights", hidden_layer, output_layer);
            add_bias_init(bias);

            // make input layer
            add_mult_op("input_mult","input","in_weights","input_mult_out");
            add_add_op("input_bias","bias","input_mult_out","input_bias_out");
            add_relu_op("input_relu","input_bias_out","input_relu_out");
            // make hidden layers except last
            string last_node("input_relu_out");
            for(int layer_num=0;layer_num<hidden_layer_count-1;layer_num++)
            {
                string numstr = to_string(layer_num);
                add_mult_op(
                    string("hidden_mult_")+numstr,
                    last_node,
                    string("hidden_weights_")+numstr,
                    string("hidden_mult_out_")+numstr);
                add_add_op(string("hidden_add_")+numstr,
                    string("hidden_mult_out_")+numstr,
                    "bias",string("hidden_add_out_")+numstr);
                add_relu_op(string("hidden_relu_")+numstr,
                    string("hidden_add_out_")+numstr,
                    string("hidden_relu_out_")+numstr);
                last_node = string("hidden_relu_out_")+numstr;
            }
            //make output layer
            add_mult_op("output_mult",last_node,"out_weights","output_mult_out");
            add_add_op("ouput_bias","bias","output_mult_out","output_bias_out");
            add_sigmoid_op("output_sigmoid","output_bias_out","output");
        }
        string toString()
        {
            return protobuf.DebugString();
        }

        void WriteToFile(string fname)
        {
            std::ofstream ofs(fname, std::ios_base::out | std::ios_base::binary);
            protobuf.SerializeToOstream(&ofs);
        }
        int GetWeightCount()
        {
            //onnx::GraphProto graph_proto = protobuf.graph();
            //int weight_count = 0;
            //for (int i = 0; i < graph_proto.initializer_size() - 1; i++)
            //{
            //    const onnx::TensorProto& tensor_proto = graph_proto.initializer(i);
            //    int tensor_size = 1;

            //    for (int j = 0; j < tensor_proto.dims_size(); j++) {
            //        tensor_size *= tensor_proto.dims(j);
            //    }

            //    weight_count += tensor_size;
            //}

            //return weight_count;

            int in_weights_count = input_layer_size * hidden_layer_size;
            int hidden_weights = (hidden_layer_size * hidden_layer_size * (num_hidden_layer - 1));
            int out_weights = hidden_layer_size * output_layer_size;

            return in_weights_count + hidden_weights + out_weights;
        }

        size_t GetByteArraySize()
        {
            return protobuf.ByteSizeLong();
        }

        uint8_t* GetByteString()
        {
            std::string serialized;
            protobuf.SerializeToString(&serialized);

            uint8_t* uint8_array = new uint8_t[serialized.length()];
            memcpy(uint8_array, serialized.data(), serialized.length());

            return uint8_array;
        }

        uint8_t* GetByteArray()
        {
            int size = protobuf.ByteSizeLong();
            uint8_t* byteArray = new uint8_t[size];
            protobuf.SerializeToArray(byteArray, size);

            //google::protobuf::io::ArrayOutputStream array_stream(byteArray, size);
            //google::protobuf::io::CodedOutputStream coded_stream(&array_stream);
            //protobuf.SerializeToCodedStream(&coded_stream);

            return byteArray;
        }

        float* ExtractWeights()
        {
            int num_weights = GetWeightCount();
            float* weights = new float[num_weights];
            onnx::GraphProto graph_proto = protobuf.graph();
            int weight_counter = 0;

            for (int i = 0; i < graph_proto.initializer_size() - 1; i++)
            {
                const onnx::TensorProto& tensor_proto = graph_proto.initializer(i);
                int tensor_size = 1;

                for (int j = 0; j < tensor_proto.dims_size(); j++) {
                    tensor_size *= tensor_proto.dims(j);
                }

                if (tensor_proto.data_type() == onnx::TensorProto_DataType_FLOAT) {

                    std::string raw_data_val = tensor_proto.raw_data();
                    const char* val = raw_data_val.c_str();

                    for (int k = 0; k < tensor_size; k++) {
                        weights[weight_counter] = tensor_proto.float_data(k);
                        weight_counter++;
                    }
                }
            }


            return weights;

        }
        void SetWeights(float* weights)
        {
            onnx::GraphProto* graph_proto = protobuf.mutable_graph();
            int weight_counter = 0;

            for (int i = 0; i < graph_proto->initializer_size() - 1; i++)
            {
                onnx::TensorProto tensor_proto_og = graph_proto->initializer(i);
                onnx::TensorProto* tensor_proto = graph_proto->mutable_initializer(i);
                int tensor_size = 1;

                for (int j = 0; j < tensor_proto->dims_size(); j++) {
                    tensor_size *= tensor_proto->dims(j);
                }

                if (tensor_proto->data_type() == onnx::TensorProto_DataType_FLOAT) {

                    for (int k = 0; k < tensor_size; k++) {
                        tensor_proto->set_float_data(k, weights[weight_counter]);
                        weight_counter++;
                    }

                }
            }

        }
        
    
    };

   
}

#pragma once
#include <fstream>
#include <onnx.pb.h>

using namespace std;

namespace onxb
{
    class MLP
    {
    private:
        onnx::ModelProto protobuf;
        int input_layer_size;
        int output_layer_size;
        

        onnx::TensorProto*  add_weights_init(string name,int depth,int width)
        {
            onnx::TensorProto* node = protobuf.mutable_graph()->add_initializer();
            node->mutable_name()->assign(name);
            node->add_dims(depth); //depth
            node->add_dims(width); // width
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
            input_layer_size(input_layer),output_layer_size(output_layer)
        {
            protobuf.clear_graph();
            protobuf.mutable_graph()->set_name("MLP Graph");
            
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
            
            
           
            //make weights nodes
            add_weights_init("in_weights",1,input_layer);
            for(int layer_num=0;layer_num<hidden_layer_count;layer_num++)
            {
                add_weights_init("hidden_weights_"+layer_num,1,
                hidden_layer);
            }
            add_weights_init("out_weights",1,output_layer);
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
        
    
    };
}

#pragma once
#include <onnx.pb.h>

using namespace std;

namespace onxb
{
    class MLP
    {
    private:
        onnx::ModelProto protobuf;
        int input_layer_size;
        int output_layerSize;
        

        onnx::TensorProto*  add_weights_init(string name,int depth,int width)
        {
            onnx::TensorProto* node = protobuf.mutable_graph()->add_initializer();
            node->mutable_name()->assign(name);
            node->add_dims(depth); //depth
            node->add_dims(width); // width
            //data will be set when we run the MLP
            return node;
        }

        onnx::TensorProto*  add_bias_init(float bias)
        {
            onnx::TensorProto* node = protobuf.mutable_graph()->add_initializer();
            node->mutable_name()->assign("bias");
            node->add_float_data(bias);
            return node;
        }

        void add_mult_op(const char*  name, const char* input1, const char* input2,
            const char* output)
        {
            onnx::NodeProto* node = protobuf.mutable_graph()->add_node();
            node->mutable_name()->assign(name);
            node->add_input(input1);
            node->add_input(input2);
            node->add_output(output);
            node->mutable_op_type()->assign("MatMul");
        }

        void add_add_op(const char*  name, const char* input1, const char* input2,
            const char* output)
        {
            onnx::NodeProto* node = protobuf.mutable_graph()->add_node();
            node->mutable_name()->assign(name);
            node->add_input(input1);
            node->add_input(input2);
            node->add_output(output);
            node->mutable_op_type()->assign("Add");
        }

        void add_relu_op(const char*  name, const char* input1,
            const char* output)
        {
            onnx::NodeProto* node = protobuf.mutable_graph()->add_node();
            node->mutable_name()->assign(name);
            node->add_input(input1);
            node->add_output(output);
            node->mutable_op_type()->assign("Relu");
        }

        
        
    public:
       

        MLP(int input_layer,int hidden_layer,int hidden_layer_count,int output_layer,
            float bias):
            input_layer_size(input_layer),output_layerSize(output_layer)
        {
            protobuf.clear_graph();
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
            add_mult_op("input_mult","input_mult","in_weights","input_bias");
            add_add_op("input_bias","input_add","bias","input_relu");
            add_relu_op("input_relu","input_relu","hidden_mult_0");
            // make hidden layers except last
            string last_node("input_relu");
            for(int layer_num=0;layer_num<hidden_layer_count-1;layer_num++)
            {
                string current_mult("hidden_mult_"+layer_num);
                string current_add("hidden_add_"+layer_num);
                string current_relu("hidden_relu_"+layer_num);
                string current_weights("hidden_weights_"+layer_num);
                string next_mult("hidden_mult_"+layer_num);
                add_mult_op(current_mult.c_str(),
                    last_node.c_str(),current_weights.c_str(),
                    current_add.c_str());
                add_add_op(current_mult.c_str(),current_add.c_str(),
                    "bias",current_relu.c_str());
                add_relu_op(current_relu.c_str(),current_add.c_str(),
                    next_mult.c_str());
                last_node = current_mult;
            }
        }
        string toString()
        {
            return protobuf.DebugString();
        }
        
    
    };
}

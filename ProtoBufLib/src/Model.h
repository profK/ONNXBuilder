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

        void add_relu_op(const char*  name, const char* input1, const char* input2,
            const char* output)
        {
            onnx::NodeProto* node = protobuf.mutable_graph()->add_node();
            node->mutable_name()->assign(name);
            node->add_input(input1);
            node->add_input(input2);
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
            add_weights_init("hidden_weights",hidden_layer_count,
                hidden_layer);
            add_weights_init("out_weights",1,output_layer);
            add_bias_init(bias);
            
        }
        string toString()
        {
            return protobuf.DebugString();
        }
        
    
    };
}

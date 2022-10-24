
// TODO: write your library functions here

// This puts a C ABI callable front end on the ONNX Builder
#include "CAPI.h"
#include "Model.h"


namespace onxb
{
    struct ModelStruct : OpaqueModel
    {
        MLP* _mlp;

        ModelStruct(int input_layer,int hidden_layer,int hidden_layer_count,int output_layer,
                float bias)
        {
            _mlp = new MLP(input_layer, hidden_layer, hidden_layer_count, output_layer,
                bias);
        }

        ~ModelStruct()
        {
            delete _mlp;
        }
    };
    
    OpaqueModel& MakeMLP(int input_layer,int hidden_layer,int hidden_layer_count,int output_layer,
                float bias)
    {
        return * new ModelStruct(input_layer,hidden_layer,hidden_layer_count,output_layer, bias);
    }
    
    int GetWeightCount(OpaqueModel& model)
    {
        //TODO
        return 0;
    }
    float* ExtractWeights(OpaqueModel& model)
    {
        //todo
        return nullptr;
    }
    void SetWeights(OpaqueModel& model, float* weights)
    {
        //TODO
    }
    
}


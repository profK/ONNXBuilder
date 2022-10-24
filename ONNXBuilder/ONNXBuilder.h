#pragma once

extern "C" {
// this is a super structure to hide the details of the model struct
    
    struct OpaqueModel
    {
    
    };
    
    OpaqueModel& MakeMLP(int input_layer,int hidden_layer,int hidden_layer_count,int output_layer,
                float bias);
    int  GetWeightCount(OpaqueModel& model);
    float* ExtractWeights(OpaqueModel& model);
    void  SetWeights(OpaqueModel& model, float* weights);
    
}

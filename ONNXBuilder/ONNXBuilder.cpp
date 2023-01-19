
// TODO: write your library functions here

// This puts a C ABI callable front end on the ONNX Builder

#include "Model.h"
#include "ONNXBuilder.h"



    struct ModelStruct : OpaqueModel
    {
        onxb::MLP* _mlp;

        ModelStruct(int input_layer,int hidden_layer,int hidden_layer_count,int output_layer,
                float bias)
        {
            _mlp = new onxb::MLP(input_layer, hidden_layer, hidden_layer_count, output_layer,
                bias);
        }

        ModelStruct()
        {
            _mlp = nullptr;
        }

        ~ModelStruct()
        {
            delete _mlp;
        }

        float* getWeights()
        {
            return _mlp->ExtractWeights();
        }

        int getWeightCount()
        {
            return _mlp->GetWeightCount();
        }

        void setWeights(float* weights)
        {
            _mlp->SetWeights(weights);
        }

        string toString()
        {
            return _mlp->toString();
        }

        int* getByteArray()
        {
            return _mlp->GetByteArray();
        }
    };
    
    OpaqueModel& MakeMLP(int input_layer,int hidden_layer,int hidden_layer_count,int output_layer,
                float bias)
    {
        return * new ModelStruct(input_layer,hidden_layer,hidden_layer_count,output_layer, bias);
    }
    
    int GetWeightCount(OpaqueModel* model)
    {
        ModelStruct* mTemp = (ModelStruct*)model;
        return mTemp->getWeightCount();
    }
    float* ExtractWeights(OpaqueModel* model)
    {
        ModelStruct* mTemp = (ModelStruct*)model;
        return mTemp->getWeights();
    }
    void SetWeights(OpaqueModel* model, float* weights)
    {
        ModelStruct* mTemp = (ModelStruct*)model;
        mTemp->setWeights(weights);

        model = (OpaqueModel*)mTemp;
    }

    int* GetByteArray(OpaqueModel* model)
    {
        ModelStruct* mTemp = (ModelStruct*)model;
        return mTemp->getByteArray();
    }



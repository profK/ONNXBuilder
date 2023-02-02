
// TODO: write your library functions here

// This puts a C ABI callable front end on the ONNX Builder

#include "Model.h"
#include "ONNXBuilder.h"

using namespace onxb;
using namespace google::protobuf;


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

        uint8_t* getByteArray()
        {
            return _mlp->GetByteArray();
        }

        size_t getByteArraySize()
        {
            return _mlp->GetByteArraySize();
        }

        void writeToFile(string fname)
        {
            _mlp->WriteToFile(fname);
        }
    };

    string GetDebugFromFile(string fname)
    {
        onnx::ModelProto model;
        fstream input(fname, ios::in | ios::binary);
        if (!model.ParseFromIstream(&input)) {
            return "Failed to parse model.";
        }
        else
        {

           return model.DebugString();
        }
    }
    
    OpaqueModel& MakeMLP(int input_layer,int hidden_layer,int hidden_layer_count,int output_layer,
                float bias)
    {
        return * new ModelStruct(input_layer,hidden_layer,hidden_layer_count,output_layer, bias);
    }

    string GetDebugString(OpaqueModel* model)
    {
        ModelStruct* mTemp = (ModelStruct*)model;
        return mTemp->toString();
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

    int GetByteArraySize(OpaqueModel* model)
    {
        ModelStruct* mTemp = (ModelStruct*)model;
        return (int)mTemp->getByteArraySize();
    }

    uint8_t* GetByteArray(OpaqueModel* model)
    {
        ModelStruct* mTemp = (ModelStruct*)model;
        return mTemp->getByteArray();
    }

    void WriteToFile(OpaqueModel* model)
    {
        ModelStruct* mTemp = (ModelStruct*)model;
        mTemp->writeToFile("newMlp.onnx");
    }



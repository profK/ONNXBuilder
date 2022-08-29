#include <fstream>
#include <google/protobuf/message.h>

#include "../ProtoBufLib/src/Model.h"
#include "gtest/gtest.h"

#include <google/protobuf/util/json_util.h>

using namespace onxb;
using namespace google::protobuf;

TEST(BasicTests, DumpFileModel)
{
    onnx::ModelProto model;
    {
        // Read the existing address book.
        fstream input("mlp.onnx", ios::in | ios::binary);
        if (!model.ParseFromIstream(&input)) {
            cerr << "Failed to parse model." << endl;
        } else
        {
            
            cout << model.DebugString() << endl;
        }
    }
}
TEST(BasicTests, DumpSyntheticModel)
{
    cout << "Dumping synthetic onnx" << endl;
    MLP testMlp(10,12,2,4,0.7f);
    cout << testMlp.toString()<<endl;
    testMlp.WriteToFile("testMlp.onnx");
}

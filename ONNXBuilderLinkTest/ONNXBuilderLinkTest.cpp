#include "Model.h"
#include "ONNXBuilder.h"

using namespace onxb;
using namespace::std;

int main(int argc, char* argv[])
{
    MLP testMlp(10, 12, 2, 4, 0.7f);
    /*cout << testMlp.toString() << endl;
    cout << "-------" << endl;*/

    int weight_count = testMlp.GetWeightCount();
    cout << "Weight Count:" << weight_count << endl;

    float* weights = testMlp.ExtractWeights();
    for (int i = 0; i < weight_count; i++)
        cout << weights[i] << " ";
    cout << endl;

    float* new_weights = new float[weight_count];

    for (int i = 0; i < weight_count; i++)
        new_weights[i] = float(i) / 100;

    testMlp.SetWeights(new_weights);

    float* weights2 = testMlp.ExtractWeights();

    cout << "Updated Weights" << endl;
    for (int i = 0; i < weight_count; i++)
        cout << weights2[i] << " ";
    cout << endl;

    //onxb::MLP mlp = onnx::MLP(10, 12, 2, 10, 0.07);
    //OpaqueModel& mlp = MakeMLP(10,12,2,10,0.07);
    return 0;
}

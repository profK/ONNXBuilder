// LinkTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "../ONNXBuilder/ONNXBuilder.h"

using namespace std;

int main()
{
    OpaqueModel& mlp = MakeMLP(10, 3, 12, 10, 0.07);

    int weight_count = GetWeightCount(&mlp);

    float* weights = ExtractWeights(&mlp);
    for (int i = 0; i < weight_count; i++)
        cout << weights[i] << " ";
    cout << endl;

    float* new_weights = new float[weight_count];
    for (int i = 0; i < weight_count; i++)
        new_weights[i] = float(i) / 100;
    SetWeights(&mlp, new_weights);

    float* weights2 = ExtractWeights(&mlp);
    for (int i = 0; i < weight_count; i++)
        cout << weights2[i] << " ";
    cout << endl;

    int* byteArray = GetByteArray(&mlp);
    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

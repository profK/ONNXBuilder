// LinkTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include "../ONNXBuilder/ONNXBuilder.h"

using namespace std;

bool AreArraysEqual(const uint8_t* array1, const uint8_t* array2, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        if (array1[i] != array2[i])
        {
            return false;
        }
    }

    return true;
}

int main()
{

    //cout << GetDebugFromFile("mlp.onnx") << endl;

    //cout << "-----------------------------" << endl;

   OpaqueModel& mlp = MakeMLP(10, 12, 2, 4, 0.07);

   int weight_count = GetWeightCount(&mlp);

   cout << "WeightCount: " << weight_count << endl;

   float* weights = ExtractWeights(&mlp);
    for (int i = 0; i < weight_count; i++)
        cout << weights[i] << " ";
    cout << endl;


    float* new_weights = new float[weight_count];
    for (int i = 0; i < weight_count; i++)
        new_weights[i] = float(i) / 1000;
    SetWeights(&mlp, new_weights);

    float* weights2 = ExtractWeights(&mlp);
    for (int i = 0; i < weight_count; i++)
        cout << weights2[i] << " ";
    cout << endl;

    int byteArraySize = GetByteArraySize(&mlp);

    cout << "ByteArraySize: " << byteArraySize << endl;

    //uint8_t* byteArray = GetByteArray(&mlp);
    //uint8_t* byteArrayString = GetByteString(&mlp);

    //cout << AreArraysEqual(byteArray, byteArrayString, byteArraySize) << endl;

    //WriteToFile(&mlp);
    //cout << GetDebugString(&mlp) << endl;

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

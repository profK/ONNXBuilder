#include "Model.h"
#include "ONNXBuilder.h"

int main(int argc, char* argv[])
{
    onxb::OpaqueModel& mlp = onxb::MakeMLP(10,12,2,10,0.07);
    return 0;
}

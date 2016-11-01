
#include <modlm/modlm-train.h>
#include <dynet/init.h>

using namespace modlm;

int main(int argc, char** argv) {
    dynet::initialize(argc, argv);
    ModlmTrain train;
    return train.main(argc, argv);
}

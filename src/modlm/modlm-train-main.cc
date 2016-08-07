
#include <modlm/modlm-train.h>
#include <cnn/init.h>

using namespace modlm;

int main(int argc, char** argv) {
    cnn::initialize(argc, argv);
    ModlmTrain train;
    return train.main(argc, argv);
}

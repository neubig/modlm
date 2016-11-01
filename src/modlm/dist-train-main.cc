
#include <modlm/dist-train.h>
#include <dynet/init.h>

using namespace modlm;

int main(int argc, char** argv) {
    DistTrain train;
    return train.main(argc, argv);
}

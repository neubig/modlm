#include <fstream>
#include <modlm/dist-factory.h>
#include <modlm/dist-ngram.h>
#include <modlm/dist-uniform.h>
#include <modlm/dist-unk.h>
#include <modlm/dist-one-hot.h>
#include <modlm/sentence.h>
#include <modlm/macros.h>
#include <modlm/input-file-stream.h>


using namespace std;
using namespace modlm;

DistPtr DistFactory::create_dist(const std::string & sig) {
  if(sig.substr(0, 5) == "ngram") {
    return DistPtr(new DistNgram(sig));
  } else if(sig == "uniform") {
    return DistPtr(new DistUniform(sig));
  } else if(sig == "unk") {
    return DistPtr(new DistUnk(sig));
  } else if(sig == "onehot") {
    return DistPtr(new DistOneHot(sig));
  } else {
    THROW_ERROR("Bad distribution signature");
  }
}

DistPtr DistFactory::from_file(const std::string & file_name, DictPtr dict) {
  InputFileStream in(file_name);
  if(!in) THROW_ERROR("Could not open " << file_name);
  string line;
  if(!getline(in, line)) THROW_ERROR("Premature end of file");
  DistPtr ret = DistFactory::create_dist(line);
  ret->read(dict, in);
  return ret;
}

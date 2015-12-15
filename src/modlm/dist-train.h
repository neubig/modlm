#pragma once


#include <string>
#include <boost/program_options.hpp>

namespace modlm {

class DistTrain {

public:
  DistTrain() { }

  int main(int argc, char** argv);
  
protected:

  boost::program_options::variables_map vm_;

  // Variable settings
  std::string model_out_file_;
  std::string train_file_;

};

}

#pragma once

#include <string>
#include <vector>

namespace modlm {

class InterpProbs {

public:
  InterpProbs() { }

  int main(int argc, char** argv);
  
protected:

  // Variable settings
  std::string prob_out_file_;
  std::vector<std::string> train_files_;
  std::vector<std::string> test_files_;

};

}

#pragma once

#include <modlm/dist-base.h>
#include <modlm/sentence.h>

namespace modlm {

class DistFactory {

public:
  static DistPtr create_dist(const std::string & sig);
  static DistPtr from_file(const std::string & file_name, DictPtr dict);

};

}

#pragma once

#include <vector>
#include <unordered_map>
#include <modlm/aggregate-data.h>

namespace modlm {

class Whitener {
public:
  Whitener(const std::string & type, float epsilon) : epsilon_(epsilon), type_(type) { }
  // Find the transformation matrix for whitening in Eigen column-major format
  void calc_matrix(const AggregateData & data);
  // Perform whitening
  void whiten(AggregateData & data);

protected:
  std::string type_;
  float epsilon_;
  std::vector<float> mean_vec_, rotation_vec_;

};

}

#pragma once

#include <vector>
#include <unordered_map>
#include <modlm/training-data.h>

namespace modlm {

typedef std::pair<TrainingContext, std::vector<std::pair<TrainingTarget, int> > > TrainingInstance;
typedef std::vector<TrainingInstance> TrainingData;

class Whitener {
public:
  Whitener(const std::string & type, float epsilon) : epsilon_(epsilon), type_(type) { }
  // Find the transformation matrix for whitening in Eigen column-major format
  void calc_matrix(const TrainingData & data);
  // Perform whitening
  void whiten(TrainingData & data);

protected:
  std::string type_;
  float epsilon_;
  std::vector<float> mean_vec_, rotation_vec_;

};

}

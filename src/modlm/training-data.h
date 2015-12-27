#pragma once

#include <vector>
#include <modlm/sentence.h>

// A training context, where:
// * first is a set of dense features 
// * second is a set of word ids
typedef std::pair<std::vector<float>, std::vector<modlm::WordId> > TrainingContext;

// A training target, where:
// * first is a dense vector of distributions
// * second is a sparse vector of distributions
typedef std::pair<std::vector<float>, std::vector<std::pair<int, float> > > TrainingTarget;

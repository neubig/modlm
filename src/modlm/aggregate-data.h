#pragma once

#include <vector>
#include <modlm/sentence.h>

// A training target, where:
// * first is a dense vector of distributions
// * second is a sparse vector of distributions
typedef std::pair<std::vector<float>, std::vector<std::pair<int, float> > > DistTarget;
typedef std::pair<int, std::vector<std::pair<int, float> > > IndexedDistTarget;

// A training context, where:
// * first is a set of dense features 
// * second is a set of word ids
typedef std::pair<std::vector<float>, std::vector<modlm::WordId> > AggregateContext;
typedef std::pair<int, std::vector<modlm::WordId> > IndexedAggregateContext;

// A set of aggregate training instances
typedef std::pair<AggregateContext, std::vector<std::pair<DistTarget, int> > > AggregateInstance;
typedef std::vector<AggregateInstance> AggregateData;

// A set of aggregate training instances
typedef std::pair<IndexedAggregateContext, std::vector<std::pair<IndexedDistTarget, int> > > IndexedAggregateInstance;
typedef std::vector<IndexedAggregateInstance> IndexedAggregateData;

#include <modlm/dist-unk.h>
#include <modlm/macros.h>

using namespace modlm;

DistUnk::DistUnk(const std::string & sig) : DistBase(sig) {
  if(sig != "unk")
    THROW_ERROR("Bad signature: " << sig);
}

void DistUnk::calc_word_dists(const Sentence & ctxt,
                             const Sentence & wids,
                             float uniform_prob,
                             float unk_prob,
                             bool leave_one_out,
                             std::vector<TrainingTarget> & trgs,
                             int & dense_offset,
                             int & sparse_offset) const {
  for(size_t i = 0; i < wids.size(); i++) 
    if(wids[i] == 0)
      trgs[i].first[dense_offset] = unk_prob;
  dense_offset++;
}

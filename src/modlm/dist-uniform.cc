#include <modlm/dist-uniform.h>
#include <modlm/macros.h>

using namespace modlm;

DistUniform::DistUniform(const std::string & sig) : DistBase(sig) {
  if(sig != "uniform")
    THROW_ERROR("Bad signature: " << sig);
}

void DistUniform::calc_word_dists(const Sentence & ngram,
                                  float uniform_prob,
                                  float unk_prob,
                                  DistTarget & trg,
                                  int & dense_offset,
                                  int & sparse_offset) const {
  trg.first[dense_offset++] = (*ngram.rbegin() == 0 ? uniform_prob * unk_prob : uniform_prob);
}

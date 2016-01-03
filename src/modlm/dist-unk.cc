#include <modlm/dist-unk.h>
#include <modlm/macros.h>

using namespace modlm;

DistUnk::DistUnk(const std::string & sig) : DistBase(sig) {
  if(sig != "unk")
    THROW_ERROR("Bad signature: " << sig);
}

void DistUnk::calc_word_dists(const Sentence & ngram,
                             float uniform_prob,
                             float unk_prob,
                             DistTarget & trg,
                             int & dense_offset,
                             int & sparse_offset) const {
  trg.first[dense_offset++] = (*ngram.rbegin() == 0 ? unk_prob : 0.f);
}

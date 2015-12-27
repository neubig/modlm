#pragma once

#include <iostream>
#include <string>
#include <modlm/sentence.h>
#include <modlm/dist-base.h>

namespace modlm {

class Counts;
typedef std::shared_ptr<Counts> CountsPtr;

// A class for the n-gram distribution
class DistNgram : public DistBase {

public:

  // Signature should be of the form
  // 1) ngram
  // 2) comb/split/all: where "comb" means combine together counts according to
  //    a heuristic, split means calculate each distribution separately, and all
  //    means to use both
  // 3) lin/wb/mkn: where "lin" means linear, "wb" means witten bell, and "mkn"
  //    means modified kneser ney
  DistNgram(const std::string & sig);
  virtual ~DistNgram() { }

  // Get the signature of this class that uniquely identifies it for loading
  // at test time. In other words, the signature can collapse any information
  // only needed at training time.
  virtual std::string get_sig() const override;

  // Add stats from one sentence at training time for count-based models
  virtual void add_stats(const Sentence & sent) override;

  // Perform finalization on stats
  virtual void finalize_stats() override;

  // Get the number of ctxtual features we can expect from this model
  virtual size_t get_ctxt_size() const override;
  // And calculate these features
  virtual void calc_ctxt_feats(const Sentence & ctxt, WordId held_out_wid, float * feats_out) const override;

  // Get the number of distributions we can expect from this model
  virtual size_t get_dense_size() const override { return counts_.size(); }
  virtual size_t get_sparse_size() const override { return 0; }
  // And calculate these features given ctxt, for words wids. uniform_prob
  // is the probability assigned in unknown ctxts. leave_one_out indicates
  // whether we should subtract one from the counts for cross-validation.
  // prob_out is the output.
  virtual void calc_word_dists(const Sentence & ctxt,
                               const Sentence & wids,
                               float uniform_prob,
                               bool leave_one_out,
                               std::vector<TrainingTarget> & trgs,
                               int & dense_offset,
                               int & sparse_offset) const override;

  // Read/write model. If dict is null, use numerical ids, otherwise strings.
  virtual void write(DictPtr dict, std::ostream & str) const override;
  virtual void read(DictPtr dict, std::istream & str) override;

  // Create the context
  static Sentence calc_ctxt(const Sentence & in, int pos, const Sentence & ctxid);

protected:


  std::vector<CountsPtr> counts_;
  std::vector<int> ctxt_pos_;
  std::string use_which_, smoothing_;

};

}


#include <boost/range/algorithm/max_element.hpp>
#include <boost/range/algorithm/min_element.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/irange.hpp>
#include <modlm/macros.h>
#include <modlm/dist-ngram.h>
#include <modlm/counts.h>

using namespace std;
using namespace modlm;

// Signature should be of the form
// 1) ngram
// 2) lin/mabs/mkn: where "lin" means linear, "mabs" means modified absolute
//    discounting, and "mkn" means modified kneser ney
DistNgram::DistNgram(const std::string & sig) : DistBase(sig) {
  // Split and sanity check signature
  std::vector<std::string> strs;
  boost::split(strs,sig,boost::is_any_of("_"));
  if(strs[0] != "ngram" || strs.size() < 2)
    THROW_ERROR("Bad signature in DistNgram: " << sig);
  // Get smoothing type
  smoothing_ = strs[1];
  if(smoothing_ != "lin" && smoothing_ != "mabs" && smoothing_ != "mkn")
    THROW_ERROR("Unimplemented smoothing type in signature");
  // Get the rest of the ctxt
  if(sig != "ngram") {
    for(auto i : boost::irange(2, (int)strs.size()))
      ctxt_pos_.push_back(stoi(strs[i]));
    if(ctxt_pos_.size() != 0) {
      ctxt_len_ = *boost::max_element(ctxt_pos_);
      if(*boost::min_element(ctxt_pos_) < 1)
        THROW_ERROR("Negative ctxt id in signature: " << sig); 
    }
  }
  // Create the counts, appropriately smoothed
  smoothing_ = strs[1];
  counts_.resize(ctxt_pos_.size()+1);
  if(smoothing_ == "lin") {
    for(auto i : boost::irange(0, (int)ctxt_pos_.size()+1))
      counts_[i].reset(new Counts);
  } else if(smoothing_ == "mabs") {
    for(auto i : boost::irange(0, (int)ctxt_pos_.size()+1))
      counts_[i].reset(new CountsMabs);
  } else if(smoothing_ == "mkn") {
    for(auto i : boost::irange(0, (int)ctxt_pos_.size()))
      counts_[i].reset(new CountsMkn);
    counts_[ctxt_pos_.size()].reset(new CountsMabs);
  } else {
    THROW_ERROR("Bad smoothing type in signature: " << sig);
  }
}

std::string DistNgram::get_sig() const {
  ostringstream oss;
  oss << "ngram_" << smoothing_;
  for(auto i : ctxt_pos_) oss << '_' << i;
  return oss.str();
}

// Add stats from one sentence at training time for count-based models
void DistNgram::add_stats(const Sentence & sent) {
  for(auto i : boost::irange(0, (int)sent.size())) {
    WordId last_fallback = -1;
    WordId this_word = sent[i];
    Sentence this_ctxt = calc_ctxt(sent, i, ctxt_pos_);
    for(size_t j = counts_.size()-1; ; j--) {
      counts_[j]->add_count(this_ctxt, this_word, last_fallback);
      if(this_ctxt.size() == 0) break;
      last_fallback = *this_ctxt.rbegin();
      this_ctxt.resize(this_ctxt.size()-1);
    };
  }
}

void DistNgram::finalize_stats() {
  for(auto & cnt : counts_)
    cnt->finalize_count();
}

// Get the number of ctxtual features we can expect from this model
size_t DistNgram::get_ctxt_size() const {
  size_t ret = 0;
  for(auto & count : counts_) ret += count->get_ctxt_size();
  return ret;
}

Sentence DistNgram::calc_ctxt(const Sentence & in, int pos, const Sentence & ctxid) {
  Sentence ret = ctxid;
  for(WordId & retid : ret)
    retid = (pos-retid < 0 ? 1 : in[pos-retid]);
  return ret;
}

// And calculate these features
void DistNgram::calc_ctxt_feats(const Sentence & ctxt, float* feats_out) const {
  Sentence this_ctxt;
  int offset = 0;
  for(size_t j = 0; ; j++) {
    (*counts_[j]).calc_ctxt_feats(this_ctxt, feats_out + offset);
    offset += counts_[j]->get_ctxt_size();
    if(j >= ctxt_pos_.size()) break;
    this_ctxt.push_back(ctxt[ctxt_pos_[j]-1]);
  }
}

// And calculate these features given ctxt, for words wids. uniform_prob
// is the probability assigned in unknown ctxts.
// prob_out is output pointer, which should be incremented after writing.

void DistNgram::calc_word_dists(const Sentence & ctxt,
                                const Sentence & wids,
                                float uniform_prob,
                                float unk_prob,
                                std::vector<AggregateTarget> & trgs,
                                int & dense_offset,
                                int & sparse_offset) const {
  Sentence this_ctxt;
  for(size_t j = 0; j < counts_.size(); j++) {
    (*counts_[j]).calc_word_dists(this_ctxt, wids, uniform_prob, unk_prob, trgs, dense_offset);
    assert(j <= ctxt_pos_.size());
    if(j < ctxt_pos_.size())
      this_ctxt.push_back(ctxt[ctxt_pos_[j]-1]);
  }
}

// Read/write model. If dict is null, use numerical ids, otherwise strings.
#define DIST_NGRAM_VERSION "distngram_v1"
void DistNgram::write(DictPtr dict, std::ostream & out) const {
  out << DIST_NGRAM_VERSION << endl;
  for(const auto & count : counts_)
    count->write(dict, out);
}
void DistNgram::read(DictPtr dict, std::istream & in) {
  string line;
  if(!getline(in, line)) THROW_ERROR("Premature end at DistNgram");
  if(line != DIST_NGRAM_VERSION) THROW_ERROR("Bad version of DistNgram: " << line << endl);
  for(int cid : boost::irange(0, (int)counts_.size()))
    counts_[cid]->read(dict, in);
}


#include <boost/range/algorithm/max_element.hpp>
#include <boost/range/algorithm/min_element.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/irange.hpp>
#include <cnn/dict.h>
#include <modlm/macros.h>
#include <modlm/dist-one-hot.h>

using namespace std;
using namespace modlm;

// Signature should be of the form
// 1) onehot
DistOneHot::DistOneHot(const std::string & sig) : DistBase(sig) {
  if(sig != "onehot")
    THROW_ERROR("Bad signature in DistOneHot: " << sig);
}

std::string DistOneHot::get_sig() const {
  return "onehot";
}

// Add stats from one sentence at training time for count-based models
void DistOneHot::add_stats(const Sentence & sent) {
  for(auto w : sent) {
    auto it = mapping_.find(w);
    if(it == mapping_.end()) {
      mapping_[w] = back_mapping_.size();
      back_mapping_.push_back(w);
    }
  }
}

void DistOneHot::finalize_stats() {
}

// Get the number of ctxtual features we can expect from this model
size_t DistOneHot::get_ctxt_size() const {
  return 0;
}

// And calculate these features
void DistOneHot::calc_ctxt_feats(const Sentence & ctxt, WordId held_out_wid, float* feats_out) const {
}

// And calculate these features given ctxt, for words wids. uniform_prob
// is the probability assigned in unknown ctxts. leave_one_out indicates
// whether we should subtract one from the counts for cross-validation.
// prob_out is the output.
void DistOneHot::calc_word_dists(const Sentence & ctxt,
                                 const Sentence & wids,
                                 float uniform_prob,
                                 float unk_prob,
                                 bool leave_one_out,
                                 std::vector<TrainingTarget> & trgs,
                                 int & dense_offset,
                                 int & sparse_offset) const {
  assert(wids.size() == trgs.size());
  for(size_t i = 0; i < wids.size(); i++) {
    auto it = mapping_.find(wids[i]);
    if(it != mapping_.end())
      trgs[i].second.push_back(make_pair(sparse_offset+it->second-1, (wids[i] == 0 ? unk_prob : 1.0)));
  }
  sparse_offset += mapping_.size();
}

// Read/write model. If dict is null, use numerical ids, otherwise strings.
#define DIST_ONEHOT_VERSION "distonehot_v1"
void DistOneHot::write(DictPtr dict, std::ostream & out) const {
  out << DIST_ONEHOT_VERSION << endl;
  for(auto i : back_mapping_)
    out << dict->Convert(i) << endl;
  out << endl;
}
void DistOneHot::read(DictPtr dict, std::istream & in) {
  string line;
  if(!(getline(in, line) && line == DIST_ONEHOT_VERSION))
    THROW_ERROR("Bad format in DistOneHot");
  while(getline(in, line)) {
    if(line == "") break;
    WordId id = dict->Convert(line);
    mapping_[id] = back_mapping_.size();
    back_mapping_.push_back(id);
  }
}

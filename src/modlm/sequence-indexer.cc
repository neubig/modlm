#include <iostream>
#include <sstream>
#include <modlm/sequence-indexer.h>
#include <modlm/macros.h>

using namespace modlm;
using namespace std;

template<>
SequenceIndexer<Sentence>::SequenceIndexer(size_t len) : len_(len), byte_len_(len * sizeof(Sentence::value_type)) { }
template<>
SequenceIndexer<vector<float> >::SequenceIndexer(size_t len) : len_(len), byte_len_(len * sizeof(vector<float>::value_type)) { }

template <class Key>
inline std::string print_vec(const Key & vec) {
  ostringstream oss;
  if(vec.size()) oss << vec[0];
  for(size_t i = 1; i < vec.size(); i++)
    oss << ' ' << vec[i];
  return oss.str();
}

template <class Key>
void SequenceIndexer<Key>::add_counts(const Key & sent) {
  Key padded_sent(sent.size()+len_-1, 1), ngram(len_);
  memcpy(&padded_sent[len_-1], &sent[0], sent.size()*sizeof(WordId));
  for(size_t i = 0; i < sent.size(); i++) {
    memcpy(&ngram[0], &padded_sent[i], byte_len_);
    indx_[ngram]++;
    //indx_.update((char*)&padded_sent[i], byte_len_, 1);
  }
}

template <class Key>
void SequenceIndexer<Key>::add_count(const Key & sent) {
  if(sent.size() != len_) THROW_ERROR("Bad sequence size in add_count: " << sent.size() << " != " << len_);
  indx_[sent]++;
}

template <class Key>
int SequenceIndexer<Key>::get_index(const Key & sent, bool allow_new) {
  if(sent.size() != len_ && len_ != -1) THROW_ERROR("Bad sequence size in get_index: " << sent.size() << " != " << len_);
  auto it = indx_.find(sent);
  if(it != indx_.end()) {
    return it->second;
  } else if(allow_new) {
    int ret = indx_.size();
    indx_.insert(it, make_pair(sent, ret));
    return ret;
  } else {
    return -1;
  }
}

template <class Key>
void SequenceIndexer<Key>::build_inverse_index(std::vector<Key> & inverse) {
  inverse.resize(indx_.size());
  for(auto & kv : indx_)
    inverse[kv.second] = kv.first;
}

namespace modlm {

template class SequenceIndexer<Sentence>;
template class SequenceIndexer<vector<float> >;

}

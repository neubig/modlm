#include <modlm/ngram-indexer.h>
#include <iostream>
#include <sstream>
#include <cstring>

using namespace modlm;
using namespace std;

template <class T>
inline std::string print_vec(const std::vector<T> vec) {
  ostringstream oss;
  if(vec.size()) oss << vec[0];
  for(size_t i = 1; i < vec.size(); i++)
    oss << ' ' << vec[i];
  return oss.str();
}

void NgramIndexer::add_counts(const Sentence & sent) {
  Sentence padded_sent(sent.size()+len_-1, 1), ngram(len_);
  memcpy(&padded_sent[len_-1], &sent[0], sent.size()*sizeof(WordId));
  for(size_t i = 0; i < sent.size(); i++) {
    memcpy(&ngram[0], &padded_sent[i], byte_len_);
    indx_[ngram]++;
    //indx_.update((char*)&padded_sent[i], byte_len_, 1);
  }
}

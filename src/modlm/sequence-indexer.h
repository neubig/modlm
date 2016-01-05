#pragma once

// #include <modlm/cedar.h>
#include <unordered_map>
#include <modlm/sentence.h>
#include <modlm/hashes.h>

namespace modlm {

template <class Key>
class SequenceIndexer {
public:

  // typedef cedar::da<int, -1, -2, false> cedar_t;

  // struct value_type {
  //   value_type(int val, size_t from, size_t len, cedar_t & indx) : second(val) {
  //     if(val != cedar_t::CEDAR_NO_PATH) {
  //       first.resize(len/sizeof(WordId)+1);
  //       indx.suffix((char*)&first[0], len, from);
  //       first.resize(len/sizeof(WordId));
  //     }
  //   }
  //   Key first;
  //   int second;
  // };

  // struct iterator {
  //   public:
  //     iterator(int val, size_t from, size_t len, cedar_t & indx) : val_(val,from,len,indx), from_(from), len_(len), indx_(&indx) { } 
  //     SequenceIndexer::iterator & operator++() {
  //       val_.second = indx_->next(from_, len_);
  //       if(val_.second != cedar_t::CEDAR_NO_PATH)
  //         val_ = value_type(val_.second, from_, len_, *indx_);
  //       return *this;
  //     }
  //     SequenceIndexer::value_type & operator*() { return val_; }
  //     bool operator==(const SequenceIndexer::iterator & it) const {
  //       return val_.second==it.val_.second && from_==it.from_ && len_==it.len_ && indx_==it.indx_;
  //     }
  //     bool operator!=(const SequenceIndexer::iterator & it) const { return !(*this == it); }
  //   protected:
  //     SequenceIndexer::value_type val_;
  //     size_t from_, len_;
  //     cedar_t *indx_;
  // };

  // SequenceIndexer::iterator begin() { 
  //   size_t from = 0, len = 0;
  //   int val = indx_.begin(from,len);
  //   if(val == cedar_t::CEDAR_NO_PATH) { from = 0; len = 0; }
  //   return SequenceIndexer::iterator(val, from, len, indx_);
  // }
  // SequenceIndexer::iterator end() { return SequenceIndexer::iterator(cedar_t::CEDAR_NO_PATH, 0, 0, indx_); }

  SequenceIndexer(size_t len);
  void add_counts(const Key & sent);
  void add_count(const Key & sent);
  int get_index(const Key & sent, bool allow_new);
  void build_inverse_index(std::vector<Key> & inverse);
  
  

  std::unordered_map<Key, int> & get_index() { return indx_; }
  const std::unordered_map<Key, int> & get_index() const { return indx_; }

protected:
  size_t len_, byte_len_;
  std::unordered_map<Key, int> indx_;
  // cedar_t indx_;
};

}

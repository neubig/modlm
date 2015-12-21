#pragma once

#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <boost/algorithm/string.hpp>
#include <modlm/macros.h>
#include <modlm/sentence.h>

namespace modlm {

class Counts {

protected:
  typedef std::pair<float, std::unordered_map<WordId, int> > ContextCounts;
  typedef std::shared_ptr<ContextCounts> ContextCountsPtr;
  std::unordered_map<Sentence, ContextCountsPtr> cnts_;

public:
  Counts() { }
  virtual ~Counts() { }

  virtual void add_count(const Sentence & idx, WordId wid, WordId last_fallback) {
    for(auto id : idx)
      if(id == -1) THROW_ERROR("Illegal context");
    ContextCountsPtr & my_cnts = cnts_[idx];
    if(my_cnts.get() == NULL) my_cnts.reset(new ContextCounts);
    my_cnts->first++;
    my_cnts->second[wid]++;
  }

  virtual void finalize_count() { }

  // Calculate the ctxtual features 
  virtual void calc_ctxt_feats(const Sentence & ctxt, WordId held_out_wid, float * fl) {
    auto it = cnts_.find(ctxt);
    // Unknown
    fl[2] = 0;
    if(it == cnts_.end() || (held_out_wid != -1 && it->second->first == mod_cnt(1))) {
      fl[0] = 0.0;
      fl[1] = 0.0;
      fl[2] = 1.0;
    } else {
      assert(it->second.get() != NULL);
      if(held_out_wid == -1) {
        fl[0] = log(it->second->first);
        fl[1] = log(mod_cnt(it->second->second.size()));
      } else {
        int cnt = it->second->second[held_out_wid];
        fl[0] = log(it->second->first - mod_cnt(cnt) + mod_cnt(cnt-1));
        fl[1] = log(it->second->second.size() - (it->second->second[held_out_wid] == mod_cnt(1) ? 1 : 0));
      }
    }
  }

  // Calculate the ctxtual features 
  virtual void calc_word_dists(const Sentence & ctxt,
                               const Sentence & wids,
                               float uniform_prob,
                               bool leave_one_out,
                               std::vector<TrainingTarget> & trgs,
                               int & dense_offset) const {
    auto it = cnts_.find(ctxt);
    if(it == cnts_.end()) {
      for(size_t i = 0; i < wids.size(); i++)
        trgs[i].first[dense_offset] = uniform_prob;
    } else {
      for(size_t i = 0; i < wids.size(); i++) {
        auto wid = wids[i];
        auto it2 = it->second->second.find(wid);
        if(it2 == it->second->second.end()) {
          trgs[i].first[dense_offset] = 0.0;
        } else if(leave_one_out) {
          float act = mod_cnt(it2->second), disc = mod_cnt(it2->second-1);
          float denom = (it->second->first - act + disc);
          trgs[i].first[dense_offset] = denom == 0.0 ? 0.0 : disc / denom;
        } else {
          trgs[i].first[dense_offset] = mod_cnt(it2->second) / it->second->first;
        }
      }
    }
    dense_offset++;
  }

  virtual void write(DictPtr dict, std::ostream & out) const {
    for(const auto & cnt : cnts_) {
      out << PrintSentence(cnt.first, dict) << '\t';
      std::string prev = "";
      assert(cnt.second.get() != NULL);
      for(const auto & kv : cnt.second->second) {
        out << prev << dict->Convert(kv.first) << ' ' << kv.second; 
        prev = " ";
      }
      out << std::endl;
    }
    out << std::endl;
  }

  // Modify the count in the appropriate way
  virtual float mod_cnt(int cnt) const { return cnt; }

  virtual void read(DictPtr dict, std::istream & in) {
    std::string line;
    std::vector<std::string> strs, words;
    while(getline(in, line)) {
      if(line == "") break;
      boost::split(strs, line, boost::is_any_of("\t"));
      assert(strs.size() == 2);
      ContextCountsPtr & ptr = cnts_[ParseSentence(strs[0], dict, false)];
      assert(ptr.get() == NULL);
      ptr.reset(new ContextCounts);
      boost::split(words, strs[1], boost::is_any_of(" "));
      assert(words.size() % 2 == 0);
      for(size_t i = 0; i < words.size(); i += 2) {
        int cnt = stoi(words[i+1]);
        ptr->first += mod_cnt(cnt);
        ptr->second[dict->Convert(words[i])] = cnt;
      }
    }
  }
  
  const std::unordered_map<Sentence, ContextCountsPtr> & get_cnts() const { return cnts_; }

};

class CountsMabs : public Counts {

public:
  CountsMabs() { }
  virtual ~CountsMabs() { }

  virtual void finalize_count() override {
    discounts_.resize(4);
    std::vector<int> fofs(5);
    for(auto & kv : cnts_)
      for(auto & cnt : kv.second->second)
        if(cnt.second < fofs.size())
          fofs[cnt.second]++;
    float Y = fofs[1] / float(fofs[1] + 2*fofs[2]);
    discounts_[1] = 1 - 2.0*Y*fofs[2]/fofs[1];
    discounts_[2] = 2 - 3.0*Y*fofs[3]/fofs[2];
    discounts_[3] = 3 - 4.0*Y*fofs[4]/fofs[3];
    for(auto & kv : cnts_)
      for(auto & cnt : kv.second->second)
        kv.second->first -= (cnt.second - mod_cnt(cnt.second));
  }

  virtual float mod_cnt(int cnt) const override {
    return cnt > 0 ? cnt - discounts_[std::min(cnt,(int)discounts_.size()-1)] : 0.0;
  }
  
  virtual void write(DictPtr dict, std::ostream & out) const override {
    out << discounts_[0];
    for(size_t i = 1; i < discounts_.size(); i++)
      out << ' ' << discounts_[i];
    out << std::endl;
    Counts::write(dict, out);
  }

  virtual void read(DictPtr dict, std::istream & in) override {
    std::string line;
    if(!getline(in, line))
      THROW_ERROR("Premature end");
    std::vector<std::string> strs;
    boost::split(strs, line, boost::is_any_of(" "));
    for(const std::string & f : strs)
      discounts_.push_back(stof(f));
    Counts::read(dict, in);
  }

protected:
  std::vector<float> discounts_;

};

class CountsMkn : public CountsMabs {

public:
  CountsMkn() { }
  virtual ~CountsMkn() { }

  virtual void add_count(const Sentence & idx, WordId wid, WordId last_fallback) override {
    if(last_fallback == -1) THROW_ERROR("Passing ctxt with no last word to kneser ney");
    ContextCountsUniqPtr & my_cnts = cnts_uniq_[idx];
    if(my_cnts.get() == NULL) my_cnts.reset(new ContextCountsUniq);
    auto & my_set = my_cnts->second[wid];
    if(my_set.find(last_fallback) == my_set.end()) {
      my_cnts->first++;
      my_set.insert(last_fallback);
    }
  }

  virtual void finalize_count() override {
    for(auto & kv_uniq : cnts_uniq_) {
      auto & kv = cnts_[kv_uniq.first];
      kv.reset(new ContextCounts);
      assert(kv_uniq.second != NULL);
      for(auto & cnt : kv_uniq.second->second)
        kv->second[cnt.first] = cnt.second.size();
    }
    cnts_uniq_.clear();
    CountsMabs::finalize_count();
  }

protected:
  typedef std::pair<int, std::unordered_map<WordId, std::unordered_set<int> > > ContextCountsUniq;
  typedef std::shared_ptr<ContextCountsUniq> ContextCountsUniqPtr;
  std::unordered_map<Sentence, ContextCountsUniqPtr> cnts_uniq_;

};

typedef std::shared_ptr<Counts> CountsPtr;

}

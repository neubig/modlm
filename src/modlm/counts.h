#pragma once

#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <modlm/hashes.h>
#include <modlm/training-data.h>

namespace cnn { class Dict; }

namespace modlm {

typedef std::shared_ptr<cnn::Dict> DictPtr;

class Counts {

protected:
  typedef std::pair<float, std::unordered_map<WordId, int> > ContextCounts;
  typedef std::shared_ptr<ContextCounts> ContextCountsPtr;
  std::unordered_map<Sentence, ContextCountsPtr> cnts_;

public:
  Counts() { }
  virtual ~Counts() { }

  virtual void add_count(const Sentence & idx, WordId wid, WordId last_fallback);

  virtual void finalize_count() { }
  virtual float mod_cnt(int cnt) const { return cnt; }

  // Calculate the ctxtual features 
  virtual void calc_ctxt_feats(const Sentence & ctxt, WordId held_out_wid, float * fl);

  // Calculate the ctxtual features 
  virtual void calc_word_dists(const Sentence & ctxt,
                               const Sentence & wids,
                               float uniform_prob,
                               bool leave_one_out,
                               std::vector<TrainingTarget> & trgs,
                               int & dense_offset) const;

  virtual void write(DictPtr dict, std::ostream & out) const;
  virtual void read(DictPtr dict, std::istream & in);
  
  const std::unordered_map<Sentence, ContextCountsPtr> & get_cnts() const { return cnts_; }

};

class CountsMabs : public Counts {

public:
  CountsMabs() { }
  virtual ~CountsMabs() { }

  virtual void finalize_count() override;

  virtual float mod_cnt(int cnt) const override;
  
  virtual void write(DictPtr dict, std::ostream & out) const override;

  virtual void read(DictPtr dict, std::istream & in) override;

protected:
  std::vector<float> discounts_;

};

class CountsMkn : public CountsMabs {

public:
  CountsMkn() { }
  virtual ~CountsMkn() { }

  virtual void add_count(const Sentence & idx, WordId wid, WordId last_fallback) override;

  virtual void finalize_count() override;

protected:
  typedef std::pair<int, std::unordered_map<WordId, std::unordered_set<int> > > ContextCountsUniq;
  typedef std::shared_ptr<ContextCountsUniq> ContextCountsUniqPtr;
  std::unordered_map<Sentence, ContextCountsUniqPtr> cnts_uniq_;

};

typedef std::shared_ptr<Counts> CountsPtr;

}

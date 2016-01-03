#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <modlm/sentence.h>
#include <modlm/timer.h>
#include <modlm/aggregate-data.h>
#include <modlm/hashes.h>

namespace cnn {
  class Model;
  class Dict;
  struct Trainer;
  struct ComputationGraph;
  struct LookupParameters;
  struct Parameters;
  struct RNNBuilder;
  namespace expr {
    struct Expression;
  }
}

namespace modlm {

class Model;
typedef std::shared_ptr<cnn::Model> ModelPtr;
class Heuristic;
typedef std::shared_ptr<Heuristic> HeuristicPtr;
class Whitener;
typedef std::shared_ptr<Whitener> WhitenerPtr;
class DistBase;
typedef std::shared_ptr<DistBase> DistPtr;
typedef std::shared_ptr<cnn::Dict> DictPtr;
typedef std::shared_ptr<cnn::RNNBuilder> BuilderPtr;

// A data structure for aggregate training instances
typedef std::unordered_map<AggregateContext, std::unordered_map<DistTarget, int> > AggregateDataMap;

class ModlmTrain {
private:
  typedef std::shared_ptr<cnn::Trainer> TrainerPtr;

public:
  ModlmTrain() : num_ctxt_(0), num_dense_dist_(0), num_sparse_dist_(0), word_hist_(0), word_rep_(50), use_context_(true) { }

  TrainerPtr get_trainer(const std::string & trainer_id, float learning_rate, cnn::Model & model);

  int main(int argc, char** argv);
  
protected:

  void print_status(const std::string & strid, int epoch, float loss, std::pair<int,int> words, float percent, float elapsed);

  // *** Shared training stuff
  cnn::expr::Expression add_to_graph(const std::vector<float> & wctxt, const std::vector<WordId> & words, const std::vector<float> & wdists, const std::vector<float> & wcnts, bool dropout, cnn::ComputationGraph & cg);
  int calc_dropout_set();

  // *** Aggregate training stuff
  void train_aggregate();
  float calc_aggregate_instance(const AggregateData & inst, const std::string & strid, std::pair<int,int> words, bool update, int epoch);
  std::pair<int,int> create_aggregate_data(const std::string & file_name, AggregateDataMap & data);
  void convert_aggregate_data(const AggregateDataMap & data_map, AggregateData & data);
  cnn::expr::Expression create_aggregate_graph(const AggregateInstance & inst, std::pair<size_t,size_t> range, std::pair<int,int> & curr_words, bool dropout, cnn::ComputationGraph & cg);

  // *** Sentence-wise training stuff
  void train_sentencewise();

  // Variable settings
  Timer time_;

  int epochs_;
  std::string model_in_file_, model_out_file_;

  ModelPtr mod_;
  DictPtr dict_;
  std::vector<DistPtr> dists_;
  HeuristicPtr heuristic_;
  WhitenerPtr whitener_;
  TrainerPtr trainer_;

  cnn::LookupParameters* reps_;
  std::vector<cnn::Parameters*> Ws_;
  std::vector<cnn::Parameters*> bs_;
  cnn::Parameters* V_;
  cnn::Parameters* a_;

  float log_unk_prob_;
  int max_ctxt_, num_ctxt_, num_dense_dist_, num_sparse_dist_;
  int word_hist_, word_rep_;
  int max_minibatch_;
  int dev_epochs_, online_epochs_;
  bool use_context_, penalize_unk_;
  std::string trainer_id_, training_type_;
  float learning_rate_, rate_decay_;
  bool clipping_enabled_;
  bool hold_out_;
  float dropout_prob_, dropout_prob_decay_;
  std::vector<std::vector<unsigned> > dropout_spans_;


  std::vector<std::vector<std::string> > model_locs_;
  std::vector<std::string> train_files_, test_files_;
  std::string valid_file_;

  BuilderPtr builder_;

};

}

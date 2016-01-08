#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <cnn/expr.h>
#include <modlm/sentence.h>
#include <modlm/timer.h>
#include <modlm/training-data.h>
#include <modlm/hashes.h>
#include <modlm/sequence-indexer.h>
#include <modlm/builder-factory.h>

namespace cnn {
  class Model;
  class Dict;
  struct Trainer;
  struct ComputationGraph;
  struct LookupParameters;
  struct Parameters;
  struct RNNBuilder;
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
typedef std::pair<int, std::vector<std::pair<int,float> > > IndexedDistTarget;
typedef std::unordered_map<IndexedAggregateContext, std::unordered_map<IndexedDistTarget, int> > IndexedAggregateDataMap;

class ModlmTrain {
private:
  typedef std::shared_ptr<cnn::Trainer> TrainerPtr;

public:
  ModlmTrain() : num_ctxt_(0), num_dense_dist_(0), num_sparse_dist_(0), word_hist_(0), word_rep_(50), use_context_(true), dist_indexer_(-1), ctxt_indexer_(-1) { }

  TrainerPtr get_trainer(const std::string & trainer_id, float learning_rate, float weight_decay, cnn::Model & model);

  int main(int argc, char** argv);
  
protected:

  void print_status(const std::string & strid, std::pair<int,int> epoch, float loss, std::pair<int,int> words, float percent, float elapsed);

  // *** Create the graph

  cnn::expr::Expression add_to_graph(size_t mb_num_sent, const std::vector<float> & wctxt, const std::vector<Sentence> & ctxt_ngrams, const std::vector<float> & wdists, const std::vector<float> & wcnts, bool dropout, cnn::ComputationGraph & cg);

  template <class Data, class Instance>
  float calc_instance(const Data & data, int minibatch, bool update, std::pair<int,int> epoch, std::pair<int,int> & words);

  // *** Perform training for the whole data set

  int calc_dropout_set();

  template <class Data, class Instance>
  float calc_dataset(const Data & data, bool update, std::pair<int,int> epoch, int my_range = 0);
  

  // *** Functions to create the dataset

  template <class DataMap, class Data>
  std::pair<int,int> create_data(const std::string & file_name, DataMap & data_map, Data & data);

  template <class DataMap, class Data>
  void finalize_data(const DataMap & data_map, Data & data);

  // *** Main training loop

  template <class DataMap, class Data, class Instance>
  void perform_training();

  // *** Sanity check stuff
  void sanity_check_aggregate(const SequenceIndexer<Sentence> & my_counts, float uniform_prob, float unk_prob);

  // **** Variable settings

  Timer time_;

  int epochs_;
  std::string model_in_file_, model_out_file_;

  ModelPtr mod_;
  DictPtr dict_;
  std::vector<DistPtr> dists_;
  HeuristicPtr heuristic_;
  WhitenerPtr whitener_;
  TrainerPtr trainer_;
  BuilderSpec hidden_spec_;

  cnn::LookupParameters* reps_;
  BuilderPtr builder_;
  cnn::Parameters* V_; cnn::expr::Expression V_expr_;
  cnn::Parameters* a_; cnn::expr::Expression a_expr_;

  float log_unk_prob_;
  int max_ctxt_, num_ctxt_, num_dense_dist_, num_sparse_dist_;
  int word_hist_, word_rep_;
  int max_minibatch_;
  int dev_epochs_, online_epochs_;
  int evaluate_frequency_;
  bool use_context_, penalize_unk_;
  std::string trainer_id_, training_type_;
  float learning_rate_, rate_decay_;
  bool clipping_enabled_;
  bool hold_out_;
  float model_dropout_prob_, model_dropout_decay_;
  float node_dropout_prob_;
  float weight_decay_;
  std::vector<std::vector<unsigned> > dropout_spans_;

  std::vector<std::vector<std::string> > model_locs_;
  std::vector<std::string> train_files_, test_files_;
  std::string valid_file_;

  SequenceIndexer<std::vector<float> > dist_indexer_, ctxt_indexer_;
  std::vector<std::vector<float> > dist_inverter_, ctxt_inverter_;

};

}

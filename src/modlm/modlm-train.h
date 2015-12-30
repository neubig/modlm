#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <modlm/sentence.h>
#include <modlm/training-data.h>
#include <modlm/hashes.h>

namespace cnn {
  class Model;
  class Dict;
  struct Trainer;
  struct ComputationGraph;
  struct LookupParameters;
  struct Parameters;
  namespace expr {
    struct Expression;
  }
}

namespace modlm {

class Heuristic;
typedef std::shared_ptr<Heuristic> HeuristicPtr;
class DistBase;
typedef std::shared_ptr<DistBase> DistPtr;
typedef std::shared_ptr<cnn::Dict> DictPtr;

// A data structure for training instances
typedef std::unordered_map<TrainingContext, std::unordered_map<TrainingTarget, int> > TrainingDataMap;
typedef std::pair<TrainingContext, std::vector<std::pair<TrainingTarget, int> > > TrainingInstance;
typedef std::vector<TrainingInstance> TrainingData;

class ModlmTrain {
private:
  typedef std::shared_ptr<cnn::Trainer> TrainerPtr;

public:
  ModlmTrain() : num_ctxt_(0), num_dense_dist_(0), num_sparse_dist_(0), word_hist_(0), word_rep_(50), use_context_(true) { }

  TrainerPtr get_trainer(const std::string & trainer_id, float learning_rate, cnn::Model & model);

  int main(int argc, char** argv);
  
protected:

  std::pair<int,int> create_data(const std::vector<DistPtr> & dists, int max_ctxt, bool hold_out, const DictPtr dict, const std::string & file_name, TrainingDataMap & data);
  void convert_data(const TrainingDataMap & data_map, TrainingData & data);
  float calc_instance(const TrainingData & inst, const std::string & strid, std::pair<int,int> words, bool update, int epoch, TrainerPtr & trainer, cnn::Model & mod);
  void print_status(const std::string & strid, int epoch, float loss, std::pair<int,int> words, float percent, float elapsed);
  cnn::expr::Expression create_graph(const TrainingInstance & inst, std::pair<size_t,size_t> range, std::pair<int,int> & curr_words, cnn::Model & mod, cnn::ComputationGraph & cg);

  // Variable settings
  int epochs_;
  std::string model_in_file_, model_out_file_;

  HeuristicPtr heuristic_;

  cnn::LookupParameters* reps_;
  std::vector<cnn::Parameters*> Ws_;
  std::vector<cnn::Parameters*> bs_;
  cnn::Parameters* V_;
  cnn::Parameters* a_;

  float log_unk_prob_;
  int num_ctxt_, num_dense_dist_, num_sparse_dist_;
  int word_hist_, word_rep_;
  int max_minibatch_;
  int dev_epochs_, online_epochs_;
  bool use_context_, penalize_unk_;
  std::string trainer_;
  float learning_rate_;
  bool clipping_enabled_;

};

}

#pragma once

#include <string>
#include <boost/program_options.hpp>
#include <cnn/cnn.h>
#include <cnn/dict.h>
#include <cnn/tensor.h>
#include <cnn/expr.h>
#include <modlm/sentence.h>
#include <modlm/dist-base.h>

namespace cnn {
struct Trainer;
class Model;
}

namespace modlm {

// A data structure for training instances
typedef std::unordered_map<TrainingContext, std::unordered_map<TrainingTarget, int> > TrainingData;
typedef std::pair<TrainingContext, std::unordered_map<TrainingTarget, int> > TrainingInstance;

class ModlmTrain {
private:
  typedef std::shared_ptr<cnn::Trainer> TrainerPtr;

public:
  ModlmTrain() : num_ctxt_(0), num_dense_dist_(0), num_sparse_dist_(0), word_hist_(0), word_rep_(50), use_context_(true) { }

  TrainerPtr GetTrainer(const std::string & trainer_id, float learning_rate, cnn::Model & model);

  cnn::expr::Expression create_graph(const TrainingInstance & inst, std::pair<size_t,size_t> range, cnn::Model & mod, cnn::ComputationGraph & cg);

  int main(int argc, char** argv);
  
protected:

  int create_instances(const std::vector<DistPtr> & dists, int max_ctxt, bool hold_out, const DictPtr dict, const std::string & file_name, TrainingData & data);


  boost::program_options::variables_map vm_;

  // Variable settings
  int epochs_;
  std::string model_in_file_, model_out_file_;
  std::string train_file_;
  std::vector<std::string> test_files_;

  cnn::LookupParameters* reps_;
  std::vector<cnn::Parameters*> Ws_;
  std::vector<cnn::Parameters*> bs_;
  cnn::Parameters* V_;
  cnn::Parameters* a_;

  int num_ctxt_, num_dense_dist_, num_sparse_dist_;
  int word_hist_, word_rep_;
  bool use_context_;
  

};

}

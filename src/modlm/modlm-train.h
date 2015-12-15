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

struct TrainingInstance {
  TrainingInstance(int num_dist, int num_ctxt, int num_word) :
    wids(num_word), ctxts(num_ctxt), wdists(num_word*num_dist), wcnts(num_word) { }
  Sentence wids;
  std::vector<float> ctxts, wdists, wcnts;
};
typedef std::shared_ptr<TrainingInstance> TrainingInstancePtr;

class Vocabulary;

class ModlmTrain {
private:
  typedef std::shared_ptr<cnn::Trainer> TrainerPtr;

public:
  ModlmTrain() : num_ctxt_(0), num_dist_(0) { }

  TrainerPtr GetTrainer(const std::string & trainer_id, float learning_rate, cnn::Model & model);

  cnn::expr::Expression create_graph(const TrainingInstance & inst, cnn::Model & mod, cnn::ComputationGraph & cg);

  int main(int argc, char** argv);
  
protected:

  std::pair<int,std::vector<TrainingInstancePtr> > create_instances(const std::vector<DistPtr> & dists, int max_ctxt, const DictPtr dict, const std::string & file_name);


  boost::program_options::variables_map vm_;

  // Variable settings
  int epochs_;
  std::string model_in_file_, model_out_file_;
  std::string train_file_;
  std::vector<std::string> test_files_;

  std::vector<cnn::Parameters*> Ws_;
  std::vector<cnn::Parameters*> bs_;
  cnn::Parameters* V_;
  cnn::Parameters* a_;

  int num_ctxt_, num_dist_;
  

};

}

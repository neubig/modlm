

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/range/irange.hpp>
#include <boost/algorithm/string.hpp>
#include <cnn/expr.h>
#include <cnn/cnn.h>
#include <cnn/dict.h>
#include <cnn/training.h>
#include <modlm/modlm-train.h>
#include <modlm/macros.h>
#include <modlm/timer.h>
#include <modlm/counts.h>
#include <modlm/dist-ngram.h>
#include <modlm/dist-factory.h>

using namespace std;
using namespace modlm;
using namespace cnn::expr;
namespace po = boost::program_options;

template <class T>
inline std::string print_vec(const std::vector<T> vec) {
  ostringstream oss;
  if(vec.size()) oss << vec[0];
  for(int i : boost::irange(1, (int)vec.size()))
    oss << ' ' << vec[i];
  return oss.str();
}

Expression ModlmTrain::create_graph(const TrainingInstance & inst, cnn::Model & mod, cnn::ComputationGraph & cg) {

  // Add the data for this instance
  Expression h = input(cg, {(unsigned int)inst.ctxts.size()}, inst.ctxts);
  Expression probs = input(cg, {(unsigned int)inst.wids.size(), (unsigned int)num_dist_}, inst.wdists);
  Expression counts = input(cg, {(unsigned int)inst.wids.size()}, inst.wcnts);

  // cerr << "wids: " << print_vec(inst.wids) << endl;
  // cerr << "wcnts: " << print_vec(inst.wcnts) << endl;
  // cerr << "wdists: " << print_vec(inst.wdists) << endl;
  // cerr << "ctxts: " << print_vec(inst.ctxts) << endl;

  // Do the NN computation
  for(size_t i = 0; i < Ws_.size(); i++)
    h = tanh( parameter(cg, Ws_[i]) * h + parameter(cg, bs_[i]) );
  Expression interp = softmax( parameter(cg, V_) * h + parameter(cg, a_) );
  Expression nlprob = -log(probs * interp);
  Expression nll = transpose(counts) * nlprob;
  return nll;
}

pair<int,vector<TrainingInstancePtr> > ModlmTrain::create_instances(const vector<DistPtr> & dists, int max_ctxt, const DictPtr dict, const std::string & file_name) {

  float uniform_prob = 1.0/dict->size();
  pair<int,CountsPtr> ret(0, CountsPtr(new Counts));

  // Load counts
  {
    ifstream in(file_name);
    if(!in) THROW_ERROR("Could not open " << file_name);
    string line;
    Sentence ctxt;
    for(int i = 1; i <= max_ctxt; i++)
      ctxt.push_back(i);
    while(getline(in, line)) {
      Sentence sent = ParseSentence(line, dict, true);
      for(int i : boost::irange(0, (int)sent.size()))
        ret.second->add_count(DistNgram::calc_ctxt(sent, i, ctxt), sent[i], -1);
    }
  }

  // Create training data (num words, ctxt features, each model, true counts)
  vector<TrainingInstancePtr> instances;
  int total_words = 0;
  for(auto & cnts : ret.second->get_cnts()) {
    TrainingInstancePtr inst(new TrainingInstance(num_dist_, num_ctxt_, cnts.second->second.size()));
    int i = 0;
    for(auto & kv : cnts.second->second) {
      inst->wids[i] = kv.first;
      inst->wcnts[i] = kv.second;
      total_words += kv.second;
      i++;
    }
    int curr_dist = 0, curr_ctxt = 0;
    for(auto dist : dists) {
      dist->calc_ctxt_feats(cnts.first, -1, &inst->ctxts[curr_ctxt]);
      dist->calc_word_dists(cnts.first, inst->wids, uniform_prob, false, &inst->wdists[curr_dist]);
      curr_dist += dist->get_dist_size() * inst->wids.size();
      curr_ctxt += dist->get_ctxt_size();
    }

    instances.push_back(inst);
  } 

  return std::make_pair(total_words, instances);
}

int ModlmTrain::main(int argc, char** argv) {
  po::options_description desc("*** modlm-train (by Graham Neubig) ***");
  desc.add_options()
      ("help", "Produce help message")
      ("train_file", po::value<string>()->default_value(""), "Training file")
      ("test_file", po::value<string>()->default_value(""), "Test file")
      ("vocab_file", po::value<string>()->default_value(""), "Vocab file")
      ("dist_models", po::value<string>()->default_value(""), "Files containing the distribution models")
      // ("model_out", po::value<string>()->default_value(""), "File to write the model to")
      // ("model_in", po::value<string>()->default_value(""), "If resuming training, read the model in")
      ("epochs", po::value<int>()->default_value(300), "Number of epochs")
      ("rate_thresh",  po::value<float>()->default_value(1e-5), "Threshold for the learning rate")
      ("trainer", po::value<string>()->default_value("adam"), "Training algorithm (sgd/momentum/adagrad/adadelta/adam)")
      ("seed", po::value<int>()->default_value(0), "Random seed (default 0 -> changes every time)")
      ("learning_rate", po::value<float>()->default_value(0.1), "Learning rate")
      ("layers", po::value<string>()->default_value("50"), "Descriptor for hidden layers, e.g. 50_30")
      ("verbose", po::value<int>()->default_value(0), "How much verbose output to print")
      ;
  po::store(po::parse_command_line(argc, argv, desc), vm_);
  po::notify(vm_);   
  if (vm_.count("help")) {
      cout << desc << endl;
      return 1;
  }

  GlobalVars::verbose = vm_["verbose"].as<int>();

  // Set random seed if necessary
  int seed = vm_["seed"].as<int>();
  if(seed != 0) {
      delete cnn::rndeng;
      cnn::rndeng = new mt19937(seed);
  }

  // Other sanity checks
  try { train_file_ = vm_["train_file"].as<string>(); } catch(std::exception & e) { }
  try { boost::split(test_files_, vm_["test_file"].as<string>(), boost::is_any_of(" ")); } catch(std::exception & e) { }
  if(test_files_.size() != 1 || test_files_[0] == "") THROW_ERROR("Must specify exactly one --test_file");
  // try { model_out_file_ = vm_["model_out"].as<string>(); } catch(std::exception & e) { }
  if(!train_file_.size())
      THROW_ERROR("Must specify a training file with --train_file");
  // if(!model_out_file_.size())
  //     THROW_ERROR("Must specify a model output file with --model_out");

  // Save some variables
  epochs_ = vm_["epochs"].as<int>();

  // Read in the vocabulary if necessary
  string line;
  DictPtr dict(new cnn::Dict);
  dict->Convert("<unk>");
  dict->Convert("<s>");
  if(vm_["vocab_file"].as<string>() != "") {
    ifstream vocab_file(vm_["vocab_file"].as<string>());
    if(!(getline(vocab_file, line) && line == "<unk>" && getline(vocab_file, line) && line == "<s>"))
      THROW_ERROR("First two lines of a vocabulary file must be <unk> and <s>");
    while(getline(vocab_file, line))
      dict->Convert(line);
    dict->Freeze();
    dict->SetUnk("<unk>");
  }

  // Read in the models
  vector<string> strs;
  boost::split(strs, vm_["dist_models"].as<string>(), boost::is_any_of(" "));
  vector<DistPtr> dists;
  size_t max_ctxt = 0;
  for(auto str : strs) {
    dists.push_back(DistFactory::from_file(str, dict));
    max_ctxt = max((*dists.rbegin())->get_ctxt_len(), max_ctxt);
  }
  if(!dict->is_frozen()) {
    dict->Freeze();
    dict->SetUnk("<unk>");
  }

  for(auto dist : dists) {
    num_dist_ += dist->get_dist_size();
    num_ctxt_ += dist->get_ctxt_size();
  }

  // Read in the data
  pair<int,vector<TrainingInstancePtr> > train_inst = create_instances(dists, max_ctxt, dict, train_file_);
  pair<int,vector<TrainingInstancePtr> > test_inst = create_instances(dists, max_ctxt, dict, test_files_[0]);

  // Initialize
  cnn::Model mod;
  TrainerPtr trainer = GetTrainer(vm_["trainer"].as<string>(), vm_["learning_rate"].as<float>(), mod);

  boost::split(strs, vm_["layers"].as<string>(), boost::is_any_of(" "));
  vector<int> hidden_size;
  for(auto str : strs)
    if(str != "")
      hidden_size.push_back(stoi(str));

  int last_size = num_ctxt_;
  for(auto size : hidden_size) {
    Ws_.push_back(mod.add_parameters({(unsigned int)size, (unsigned int)last_size}));
    bs_.push_back(mod.add_parameters({(unsigned int)size}));
    last_size = size;
  }
  V_ = mod.add_parameters({(unsigned int)num_dist_, (unsigned int)last_size});
  a_ = mod.add_parameters({(unsigned int)num_dist_});

  // Train a neural network to predict the interpolation coefficients
  for(int epoch = 1; epoch < 300; epoch++) {
    random_shuffle(train_inst.second.begin(), train_inst.second.end());
    float train_loss = 0.0, test_loss = 0.0;
    for(auto inst : train_inst.second) {
      cnn::ComputationGraph cg;
      create_graph(*inst, mod, cg);
      train_loss += cnn::as_scalar(cg.forward());
      cg.backward();
      if(epoch <= 2)
        trainer->update();
    }
    if(epoch > 2)
      trainer->update();
    trainer->update_epoch();
    float train_ppl = exp(train_loss/train_inst.first);
    cout << "trn loss epoch " << epoch << ": " << train_ppl << endl;
    // Test PPL
    if(epoch % 10 == 0) {
      for(auto inst : test_inst.second) {
        cnn::ComputationGraph cg;
        create_graph(*inst, mod, cg);
        test_loss += cnn::as_scalar(cg.forward());
      }
      float test_ppl = exp(test_loss/test_inst.first);
      cout << "--- tst loss epoch " << epoch << ": " << test_ppl << endl;
    }
  }

  return 0;
}

ModlmTrain::TrainerPtr ModlmTrain::GetTrainer(const std::string & trainer_id, float learning_rate, cnn::Model & model) {
    TrainerPtr trainer;
    if(trainer_id == "sgd") {
        trainer.reset(new cnn::SimpleSGDTrainer(&model, 1e-6, learning_rate));
    } else if(trainer_id == "momentum") {
        trainer.reset(new cnn::MomentumSGDTrainer(&model, 1e-6, learning_rate));
    } else if(trainer_id == "adagrad") {
        trainer.reset(new cnn::AdagradTrainer(&model, 1e-6, learning_rate));
    } else if(trainer_id == "adadelta") {
        trainer.reset(new cnn::AdadeltaTrainer(&model, 1e-6, learning_rate));
    } else if(trainer_id == "adam") {
        trainer.reset(new cnn::AdamTrainer(&model, 1e-6, learning_rate));
    } else {
        THROW_ERROR("Illegal trainer variety: " << trainer_id);
    }
    return trainer;
}

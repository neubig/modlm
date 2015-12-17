

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

  // Dynamically create the target vectors
  vector<float> wdists(inst.second.size() * num_dist_), wcnts(inst.second.size());
  size_t pos = 0;
  for(auto & kv : inst.second) {
    memcpy(&wdists[pos * num_dist_], &kv.first[0], sizeof(float)*num_dist_);
    wcnts[pos] = kv.second;
    pos++;
  }
  // Load the targets
  Expression probs = input(cg, {(unsigned int)num_dist_, (unsigned int)inst.second.size()}, wdists);
  Expression counts = input(cg, {(unsigned int)inst.second.size()}, wcnts);

  // cerr << "wcnts: " << print_vec(wcnts) << endl;
  // cerr << "wdists: " << print_vec(wdists) << endl;

  // If not using context, it's really simple
  if(!use_context_) {
    Expression nlprob = -log(transpose(probs) * softmax( parameter(cg, a_) ) );
    Expression nll = transpose(counts) * nlprob;
    return nll;
  }

  // Add the context for this instance
  Expression h = input(cg, {(unsigned int)inst.first.first.size()}, inst.first.first);

  // Do the NN computation
  if(word_hist_ != 0) {
    vector<Expression> expr_cat(word_hist_+1);
    expr_cat[0] = h;
    for(size_t i = 0; i < inst.first.second.size(); i++)
      expr_cat[i+1] = lookup(cg, reps_, inst.first.second[i]);
    h = concatenate(expr_cat);
  }
  for(size_t i = 0; i < Ws_.size(); i++)
    h = tanh( parameter(cg, Ws_[i]) * h + parameter(cg, bs_[i]) );
  Expression interp = softmax( parameter(cg, V_) * h + parameter(cg, a_) );
  Expression nlprob = -log(transpose(probs) * interp);
  Expression nll = transpose(counts) * nlprob;
  return nll;
}

inline void calc_all_contexts(const vector<DistPtr> & dists, const Sentence & sent, int hold_out, TrainingContext & ctxt) {
   int curr_ctxt = 0;
   for(auto dist : dists) {
     dist->calc_ctxt_feats(sent, -1, &ctxt.first[curr_ctxt]);
     curr_ctxt += dist->get_ctxt_size();
   }
}

int ModlmTrain::create_instances(const vector<DistPtr> & dists, int max_ctxt, bool hold_out, const DictPtr dict, const std::string & file_name, TrainingData & data) {

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
  int total_words = 0;
  // Create and allocate the targets:
  TrainingContext ctxt;
  ctxt.first.resize(num_ctxt_);
  ctxt.second.resize(word_hist_);
  // Loop through each of the contexts
  for(auto & cnts : ret.second->get_cnts()) {
    // Calculate the words and contexts (if not holding out)
    for(int i = 0; i < word_hist_; i++)
      ctxt.second[i] = cnts.first[i];
    if(!hold_out)
      calc_all_contexts(dists, cnts.first, -1, ctxt);
    // Prepare the pointers for each word
    Sentence wids, wcnts;
    std::vector<float*> ptrs(cnts.second->second.size());
    std::vector<TrainingTarget> trgs(cnts.second->second.size(), TrainingTarget(num_dist_));
    for(auto & kv : cnts.second->second) {
      ptrs[wids.size()] = &trgs[wids.size()][0];
      wids.push_back(kv.first);
      wcnts.push_back(kv.second);
      total_words += kv.second;
    }
    // Calculate all of the distributions
    for(auto dist : dists)
      dist->calc_word_dists(cnts.first, wids, uniform_prob, hold_out, ptrs);
    // Add counts for each context
    for(int i : boost::irange(0, (int)wids.size())) {
      if(hold_out)
        calc_all_contexts(dists, cnts.first, wids[i], ctxt);
      data[ctxt][trgs[i]] += wcnts[i];
    }
  } 

  return total_words;
}

int ModlmTrain::main(int argc, char** argv) {
  po::options_description desc("*** modlm-train (by Graham Neubig) ***");
  desc.add_options()
      ("help", "Produce help message")
      ("train_file", po::value<string>()->default_value(""), "Training file")
      ("test_file", po::value<string>()->default_value(""), "Test file")
      ("vocab_file", po::value<string>()->default_value(""), "Vocab file")
      ("dist_models", po::value<string>()->default_value(""), "Files containing the distribution models")
      ("word_hist", po::value<int>()->default_value(0), "Word history length")
      ("word_rep", po::value<int>()->default_value(50), "Word representation size")
      ("hold_out", po::value<bool>()->default_value(false), "Whether to perform holding one out")
      ("use_context", po::value<bool>()->default_value(true), "If set to false, learn context-independent coefficients")
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

  // Get word history and word representation size
  word_hist_ = vm_["word_hist"].as<int>();
  word_rep_ = vm_["word_rep"].as<int>();

  // Read in the models
  vector<string> strs;
  boost::split(strs, vm_["dist_models"].as<string>(), boost::is_any_of(" "));
  vector<DistPtr> dists;
  size_t max_ctxt = word_hist_;
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

  // Use context
  use_context_ = vm_["use_context"].as<bool>();

  // Read in the data
  TrainingData train_inst, test_inst;
  int train_words = create_instances(dists, max_ctxt, vm_["hold_out"].as<bool>(), dict, train_file_, train_inst);
  int test_words  = create_instances(dists, max_ctxt, false, dict, test_files_[0], test_inst);
  dists.clear();

  // Initialize
  cnn::Model mod;
  TrainerPtr trainer = GetTrainer(vm_["trainer"].as<string>(), vm_["learning_rate"].as<float>(), mod);

  boost::split(strs, vm_["layers"].as<string>(), boost::is_any_of(" "));
  vector<int> hidden_size;
  for(auto str : strs)
    if(str != "")
      hidden_size.push_back(stoi(str));

  if(use_context_) {
    int last_size = num_ctxt_ + word_rep_ * word_hist_;
    // Add the word representation and transformation functions
    if(word_hist_ != 0)
      reps_ = mod.add_lookup_parameters(dict->size(), {(unsigned int)word_rep_});
    // Add the functions
    for(auto size : hidden_size) {
      Ws_.push_back(mod.add_parameters({(unsigned int)size, (unsigned int)last_size}));
      bs_.push_back(mod.add_parameters({(unsigned int)size}));
      last_size = size;
    }
    V_ = mod.add_parameters({(unsigned int)num_dist_, (unsigned int)last_size});
  }
  a_ = mod.add_parameters({(unsigned int)num_dist_});

  // Train a neural network to predict the interpolation coefficients
  for(int epoch = 1; epoch <= epochs_; epoch++) {
    float train_loss = 0.0, test_loss = 0.0;
    for(auto inst : train_inst) {
      cnn::ComputationGraph cg;
      create_graph(inst, mod, cg);
      train_loss += cnn::as_scalar(cg.forward());
      cg.backward();
      if(epoch <= 2)
        trainer->update();
    }
    if(epoch > 2)
      trainer->update();
    trainer->update_epoch();
    float train_ppl = exp(train_loss/train_words);
    cout << "trn loss epoch " << epoch << ": " << train_ppl << endl;
    // Test PPL
    if(epoch % 10 == 0) {
      for(auto inst : test_inst) {
        cnn::ComputationGraph cg;
        create_graph(inst, mod, cg);
        test_loss += cnn::as_scalar(cg.forward());
      }
      float test_ppl = exp(test_loss/test_words);
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

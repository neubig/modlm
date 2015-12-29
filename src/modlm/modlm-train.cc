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
#include <modlm/dict-utils.h>

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


float ModlmTrain::calc_instance(const TrainingData & data, const std::string & strid, std::pair<int,int> words, bool update, int epoch, TrainerPtr & trainer, cnn::Model & mod) {
  float loss = 0.0;
  Timer time;
  for(auto inst : data) {
    for(size_t i = 0; i < inst.second.size(); i += max_minibatch_) {
      cnn::ComputationGraph cg;
      create_graph(inst, make_pair(i, min(inst.second.size(), i+max_minibatch_)), mod, cg);
      loss += cnn::as_scalar(cg.forward());
      cg.backward();
      if(update && (online_epochs_ == -1 || epoch <= online_epochs_))
        trainer->update();
    }
  }
  if(update && online_epochs_ != -1 && epoch > online_epochs_)
    trainer->update();
  trainer->update_epoch();
  if(loss != loss)
    THROW_ERROR("Loss is not a number");
  float ppl = exp(loss/words.first);
  float ppl_nounk = exp((loss + words.second * log_unk_prob_)/words.first);
  float elapsed = time.Elapsed();
  float wps = words.first / elapsed;
  cout << strid << " epoch " << epoch << ": ppl=" << ppl << "   (";
  if(penalize_unk_) cout << "ppl_nounk=" << ppl_nounk << ", ";
  cout << "s=" << elapsed << ", wps=" << wps << ")" << endl;
  return loss;
}

Expression ModlmTrain::create_graph(const TrainingInstance & inst, pair<size_t,size_t> range, cnn::Model & mod, cnn::ComputationGraph & cg) {

  // Dynamically create the target vectors
  int num_dist = num_dense_dist_ + num_sparse_dist_;
  int num_words = (range.second - range.first);
  vector<float> wdists(num_words * num_dist, 0.0), wcnts(num_words);
  size_t pos = 0, ptr = 0;
  for(auto & kv : inst.second) {
    if(pos >= range.first && pos < range.second) {
      memcpy(&wdists[ptr], &kv.first.first[0], sizeof(float)*num_dense_dist_);
      ptr += num_dense_dist_;
      for(auto & elem : kv.first.second)
        wdists[ptr + elem.first] = elem.second;
      ptr += num_sparse_dist_;
      wcnts[pos-range.first] = kv.second;
    }
    pos++;
  }
  // Load the targets
  Expression probs = input(cg, {(unsigned int)num_dist, (unsigned int)num_words}, wdists);
  Expression counts = input(cg, {(unsigned int)num_words}, wcnts);

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
    vector<Expression> expr_cat;
    if(inst.first.first.size() != 0)
      expr_cat.push_back(h);
    // cerr << "wreps:";
    for(size_t i = 0; i < inst.first.second.size(); i++) {
      expr_cat.push_back(lookup(cg, reps_, inst.first.second[i]));
      // cerr << " " << inst.first.second[i];
    }
    // cerr << endl;
    h = (expr_cat.size() > 1 ? concatenate(expr_cat) : expr_cat[0]);
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
     assert(dist.get() != NULL);
     dist->calc_ctxt_feats(sent, -1, &ctxt.first[curr_ctxt]);
     curr_ctxt += dist->get_ctxt_size();
   }
}

pair<int,int> ModlmTrain::create_instances(const vector<DistPtr> & dists, int max_ctxt, bool hold_out, const DictPtr dict, const string & file_name, TrainingData & data) {

  float uniform_prob = 1.0/dict->size();
  float unk_prob = (penalize_unk_ ? uniform_prob : 1);
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
  pair<int,int> total_words(0,0);
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
    vector<float*> ptrs(cnts.second->second.size());
    vector<TrainingTarget> trgs(cnts.second->second.size(), TrainingTarget(vector<float>(num_dense_dist_), vector<pair<int,float> >()));
    for(auto & kv : cnts.second->second) {
      wids.push_back(kv.first);
      wcnts.push_back(kv.second);
      total_words.first += kv.second;
      if(kv.first == 0) total_words.second += kv.second;
    }
    // cerr << "wids: " << print_vec(wids) << endl;
    // cerr << "cnts: " << print_vec(wcnts) << endl;
    // Calculate all of the distributions
    int dense_offset = 0, sparse_offset = 0;
    for(auto dist : dists)
      dist->calc_word_dists(cnts.first, wids, uniform_prob, unk_prob, hold_out, trgs, dense_offset, sparse_offset);
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
      ("train_file", po::value<string>()->default_value(""), "One or more training files split with pipes")
      ("valid_file", po::value<string>()->default_value(""), "Validation file for tuning parameters")
      ("test_file", po::value<string>()->default_value(""), "One or more testing files split with pipes")
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
      ("max_minibatch", po::value<int>()->default_value(256), "Max minibatch size")
      ("dev_epochs", po::value<int>()->default_value(10), "Run the development set every x epochs")
      ("online_epochs", po::value<int>()->default_value(-1), "Number of epochs of online learning to perform before switching to batch (-1: only online)")
      ("penalize_unk", po::value<bool>()->default_value(true), "Whether to penalize unknown words")
      ("seed", po::value<int>()->default_value(0), "Random seed (default 0 -> changes every time)")
      ("cnn_mem", po::value<int>()->default_value(512), "Memory used by cnn in megabytes")
      ("learning_rate", po::value<float>()->default_value(0.1), "Learning rate")
      ("clipping_enabled", po::value<bool>()->default_value(true), "Whether to enable clipping or not")
      ("layers", po::value<string>()->default_value("50"), "Descriptor for hidden layers, e.g. 50_30")
      ("verbose", po::value<int>()->default_value(0), "How much verbose output to print")
      ;
  boost::program_options::variables_map vm_;
  po::store(po::parse_command_line(argc, argv, desc), vm_);
  po::notify(vm_);   
  if (vm_.count("help")) {
      cout << desc << endl;
      return 1;
  }

  // Save various settings
  GlobalVars::verbose = vm_["verbose"].as<int>();
  word_hist_ = vm_["word_hist"].as<int>();
  word_rep_ = vm_["word_rep"].as<int>();
  use_context_ = vm_["use_context"].as<bool>();
  penalize_unk_ = vm_["penalize_unk"].as<bool>();
  epochs_ = vm_["epochs"].as<int>();
  max_minibatch_ = vm_["max_minibatch"].as<int>();
  dev_epochs_ = vm_["dev_epochs"].as<int>();
  online_epochs_ = vm_["online_epochs"].as<int>();

  // Set random seed if necessary
  int seed = vm_["seed"].as<int>();
  if(seed != 0) {
    delete cnn::rndeng;
    cnn::rndeng = new mt19937(seed);
  }

  // Other sanity checks
  vector<string> train_files, test_files;
  string valid_file;
  try { boost::split(train_files, vm_["train_file"].as<string>(), boost::is_any_of("|")); } catch(exception & e) { }
  if(train_files.size() < 1 || train_files[0] == "") THROW_ERROR("Must specify at least one --train_file");
  try { valid_file = vm_["valid_file"].as<string>(); } catch(exception & e) { }
  try { boost::split(test_files, vm_["test_file"].as<string>(), boost::is_any_of("|")); } catch(exception & e) { }
  if(test_files.size() < 1 || test_files[0] == "") THROW_ERROR("Must specify at least one --test_file");

  // Temporary buffers
  string line;
  vector<string> strs;

  // Read in the vocabulary if necessary
  DictPtr dict(new cnn::Dict);
  dict->Convert("<unk>");
  dict->Convert("<s>");
  string vocab_file = vm_["vocab_file"].as<string>();
  if(vocab_file == "")
    THROW_ERROR("Must specify a vocabulary file");
  ifstream vocab_in(vocab_file);
  if(!(getline(vocab_in, line) && line == "<unk>" && getline(vocab_in, line) && line == "<s>"))
    THROW_ERROR("First two lines of a vocabulary file must be <unk> and <s>");
  while(getline(vocab_in, line))
    dict->Convert(line);
  dict->Freeze();
  dict->SetUnk("<unk>");

  // Read in the model locations. For each type of model, there must be either one model, or one
  // testing model, and then a model for each of the training folds
  vector<string> model_types;
  vector<vector<string> > model_locs;
  boost::split(model_types, vm_["dist_models"].as<string>(), boost::is_any_of(" "));
  for(auto str : model_types) {
    vector<string> my_locs;
    boost::split(my_locs, str, boost::is_any_of("|"));
    if(my_locs.size() != 1 && my_locs.size() != train_files.size() + 1)
      THROW_ERROR("When using cross-validation on the training data, must have appropriate model size.");
    model_locs.push_back(my_locs);
  }
  // Read in the the testing distributions
  vector<DistPtr> dists;
  size_t max_ctxt = word_hist_;
  for(auto & locs : model_locs) {
    DistPtr dist = DistFactory::from_file(locs[0], dict);
    dists.push_back(dist);
    max_ctxt = max(dist->get_ctxt_len(), max_ctxt);
    num_dense_dist_ += dist->get_dense_size();
    num_sparse_dist_ += dist->get_sparse_size();
    num_ctxt_ += dist->get_ctxt_size();
  }

  // Create the testing/validation instances
  TrainingData train_inst, valid_inst;
  pair<int,int> train_words(0,0), valid_words(0,0);
  vector<TrainingData> test_inst(test_files.size());
  vector<pair<int,int> > test_words(test_files.size(), pair<int,int>(0,0));
  if(valid_file != "")
    valid_words = create_instances(dists, max_ctxt, false, dict, valid_file, valid_inst);
  for(size_t i = 0; i < test_files.size(); i++)
    test_words[i]  = create_instances(dists, max_ctxt, false, dict, test_files[i], test_inst[i]);

  // Create the training instances
  for(size_t i = 0; i < train_files.size(); i++) {
    for(size_t j = 0; j < model_locs.size(); j++) {
      if(model_locs[j].size() != 1)
        dists[j] = DistFactory::from_file(model_locs[j][i+1], dict);
    }
    pair<int,int> my_words = create_instances(dists, max_ctxt, vm_["hold_out"].as<bool>(), dict, train_files[i], train_inst);
    train_words.first += my_words.first; train_words.second += my_words.second;
  }
  dists.clear();

  // Initialize
  cnn::Model mod;
  TrainerPtr trainer = GetTrainer(vm_["trainer"].as<string>(), vm_["learning_rate"].as<float>(), mod);
  trainer->clipping_enabled = vm_["clipping_enabled"].as<bool>();

  float uniform_prob = 1.0/dict->size();
  log_unk_prob_ = vm_["penalize_unk"].as<bool>() ? log(uniform_prob) : 0;

  boost::split(strs, vm_["layers"].as<string>(), boost::is_any_of(" "));
  vector<int> hidden_size;
  for(auto str : strs)
    if(str != "")
      hidden_size.push_back(stoi(str));

  int num_dist = num_sparse_dist_ + num_dense_dist_;
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
    V_ = mod.add_parameters({(unsigned int)num_dist, (unsigned int)last_size});
  }
  a_ = mod.add_parameters({(unsigned int)num_dist});

  // Train a neural network to predict the interpolation coefficients
  float last_valid = 1e99;
  for(int epoch = 1; epoch <= epochs_; epoch++) { 
    calc_instance(train_inst, "-- trn ", train_words, true, epoch, trainer, mod);
    if(valid_inst.size() != 0) {
      float valid_loss = calc_instance(valid_inst, "   vld ", valid_words, false, epoch, trainer, mod);
      if(last_valid < valid_loss) {
        trainer->eta *= 0.8;
        cout << "*** Reduced learning rate to " << trainer->eta << endl;
      }
    }
    for(size_t i = 0; i < test_inst.size(); i++) {
      ostringstream oss;
      calc_instance(test_inst[i], "   tst" + to_string(i), test_words[i], false, epoch, trainer, mod);
    }
  }

  return 0;
}

ModlmTrain::TrainerPtr ModlmTrain::GetTrainer(const string & trainer_id, float learning_rate, cnn::Model & model) {
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
    } else if(trainer_id == "rms") {
        trainer.reset(new cnn::RmsPropTrainer(&model, 1e-6, learning_rate));
    } else {
        THROW_ERROR("Illegal trainer variety: " << trainer_id);
    }
    return trainer;
}

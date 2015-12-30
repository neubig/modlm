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
#include <modlm/whitener.h>
#include <modlm/heuristic.h>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

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

void ModlmTrain::print_status(const std::string & strid, int epoch, float loss, pair<int,int> words, float percent, float elapsed) {
  float ppl = exp(loss/words.first);
  float ppl_nounk = exp((loss + words.second * log_unk_prob_)/words.first);
  float wps = words.first / elapsed;
  cout << strid << " epoch " << epoch;
  if(percent >= 0) cout << " (" << percent << "%)";
  cout << ": ppl=" << ppl << "   (";
  if(penalize_unk_ && words.second != -1) cout << "ppl_nounk=" << ppl_nounk << ", ";
  cout << "s=" << elapsed << ", wps=" << wps << ")" << endl;
}

float ModlmTrain::calc_instance(const TrainingData & data, const std::string & strid, std::pair<int,int> words, bool update, int epoch, TrainerPtr & trainer, cnn::Model & mod) {
  float loss = 0.0, print_every = 60.0, elapsed;
  float last_print = 0;
  Timer time;
  pair<int,int> curr_words(0,-1);
  int data_done = 0;
  for(auto inst : data) {
    for(size_t i = 0; i < inst.second.size(); i += max_minibatch_) {
      cnn::ComputationGraph cg;
      create_graph(inst, make_pair(i, min(inst.second.size(), i+max_minibatch_)), curr_words, update, mod, cg);
      loss += cnn::as_scalar(cg.forward());
      if(loss != loss) THROW_ERROR("Loss is not a number");
      if(update) {
        cg.backward();
        if(online_epochs_ == -1 || epoch <= online_epochs_)
          trainer->update();
      }
    }
    elapsed = time.Elapsed();
    data_done++;
    if(elapsed > last_print + print_every) {
      print_status(strid, epoch, loss, curr_words, 100.0*curr_words.first/words.first, elapsed);
      last_print += print_every;
    }
  }
  elapsed = time.Elapsed();
  print_status(strid, epoch, loss, words, -1, elapsed);
  if(update) {
    if(online_epochs_ != -1 && epoch > online_epochs_)
      trainer->update();
    trainer->update_epoch();
  }
  return loss;
}

Expression ModlmTrain::create_graph(const TrainingInstance & inst, pair<size_t,size_t> range, pair<int,int> & words, bool dropout, cnn::Model & mod, cnn::ComputationGraph & cg) {

  // Dynamically create the target vectors
  int num_dist = num_dense_dist_ + num_sparse_dist_;
  int num_words = (range.second - range.first);
  vector<float> wdists(num_words * num_dist, 0.0), wcnts(num_words);
  size_t ptr = 0;
  for(size_t pos = range.first; pos < range.second; pos++) {
    auto & kv = inst.second[pos];
    memcpy(&wdists[ptr], &kv.first.first[0], sizeof(float)*num_dense_dist_);
    ptr += num_dense_dist_;
    for(auto & elem : kv.first.second)
      wdists[ptr + elem.first] = elem.second;
    ptr += num_sparse_dist_;
    wcnts[pos-range.first] = kv.second;
    words.first += kv.second;
  }
  // Load the targets
  Expression probs = input(cg, {(unsigned int)num_dist, (unsigned int)num_words}, wdists);
  Expression counts = input(cg, {(unsigned int)num_words}, wcnts);

  // cerr << "wcnts:  " << print_vec(wcnts) << endl;
  // cerr << "wdists: " << print_vec(wdists) << endl;
  // cerr << "wctxt:  " << print_vec(inst.first.first) << endl;

  // If not using context, it's really simple
  if(!use_context_) {
    if(dropout_prob_ > 0) THROW_ERROR("dropout not implemented for no context");
    Expression nlprob = -log(transpose(probs) * softmax( parameter(cg, a_) ) );
    Expression nll = transpose(counts) * nlprob;
    return nll;
  }

  // If using heuristics, then perform heuristic smoothing
  if(heuristic_.get() != NULL) {
    vector<float> interp = heuristic_->smooth(num_dense_dist_, inst.first.first);
    Expression nlprob = -log( transpose(probs) * input(cg, {(unsigned int)num_dist}, interp) );
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
    for(size_t i = 0; i < inst.first.second.size(); i++)
      expr_cat.push_back(lookup(cg, reps_, inst.first.second[i]));
    h = (expr_cat.size() > 1 ? concatenate(expr_cat) : expr_cat[0]);
  }
  for(size_t i = 0; i < Ws_.size(); i++)
    h = tanh( parameter(cg, Ws_[i]) * h + parameter(cg, bs_[i]) );
  Expression softmax_input = parameter(cg, V_) * h + parameter(cg, a_);
  Expression interp; 
  uniform_real_distribution<float> float_distribution(0.0, 1.0);  
  if(!dropout || dropout_prob_ == 0 || float_distribution(*cnn::rndeng) >= dropout_prob_) {
    interp = softmax( softmax_input );
  } else {
    assert(dropout_spans_.size() > 0);
    uniform_int_distribution<int> int_distribution(0, (int)dropout_spans_.size()-1);  
    int dropout_dist = int_distribution(*cnn::rndeng);
    interp = exp( log_softmax( softmax_input, dropout_spans_[dropout_dist] ) );
  }
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


void ModlmTrain::convert_data(const TrainingDataMap & data_map, TrainingData & data) {
  data.resize(data_map.size());
  auto outer_it = data.begin();
  for(auto & dm : data_map) {
    outer_it->first = dm.first;
    outer_it->second.resize(dm.second.size());
    auto inner_it = outer_it->second.begin();
    for(auto & dmin : dm.second) {
      *inner_it = dmin;
      inner_it++;
    }
    outer_it++;
  }
}

pair<int,int> ModlmTrain::create_data(const vector<DistPtr> & dists, int max_ctxt, bool hold_out, const DictPtr dict, const string & file_name, TrainingDataMap & data) {

  float uniform_prob = 1.0/dict->size();
  float unk_prob = (penalize_unk_ ? uniform_prob : 1);
  pair<int,CountsPtr> ret(0, CountsPtr(new Counts));

  // Load counts
  {
    ifstream in(file_name);
    if(!in) THROW_ERROR("Could not open in create_instances: " << file_name);
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
    vector<float*> ptrs(cnts.second->cnts.size());
    vector<TrainingTarget> trgs(cnts.second->cnts.size(), TrainingTarget(vector<float>(num_dense_dist_), vector<pair<int,float> >()));
    for(auto & kv : cnts.second->cnts) {
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

inline std::vector<std::string> split_wildcarded(const std::string & str, const std::vector<std::string> & wildcards, const std::string & delimiter, bool skip_first_wildcard) {
  std::vector<std::string> ret;
  auto end = str.find("WILD");
  if(end != string::npos) {
    string left = str.substr(0, end);
    string right = str.substr(end+4);
    for(size_t i = (skip_first_wildcard ? 1 : 0); i < wildcards.size(); i++)
      ret.push_back(left+wildcards[i]+right);
  } else {
    boost::split(ret, str, boost::is_any_of(delimiter));
  }
  return ret;
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
      ("model_out", po::value<string>()->default_value(""), "File to write the model to")
      ("model_in", po::value<string>()->default_value(""), "If resuming training, read the model in")
      ("epochs", po::value<int>()->default_value(300), "Number of epochs")
      ("rate_thresh",  po::value<float>()->default_value(1e-5), "Threshold for the learning rate")
      ("trainer", po::value<string>()->default_value("adam"), "Training algorithm (sgd/momentum/adagrad/adadelta/adam)")
      ("max_minibatch", po::value<int>()->default_value(256), "Max minibatch size")
      ("online_epochs", po::value<int>()->default_value(-1), "Number of epochs of online learning to perform before switching to batch (-1: only online)")
      ("penalize_unk", po::value<bool>()->default_value(true), "Whether to penalize unknown words")
      ("wildcards", po::value<string>()->default_value(""), "Wildcards in model/data names for cross validation")
      ("cnn_seed", po::value<int>()->default_value(0), "Random seed (default 0 -> changes every time)")
      ("cnn_mem", po::value<int>()->default_value(512), "Memory used by cnn in megabytes")
      ("learning_rate", po::value<float>()->default_value(0.1), "Learning rate")
      ("rate_decay", po::value<float>()->default_value(1.0), "How much to decay learning rate when validation likelihood decreases")
      ("whiten", po::value<string>()->default_value(""), "Type of whitening (mean/pca/zca)")
      ("whiten_eps", po::value<float>()->default_value(0.01), "Regularization for whitening")
      ("dropout_models", po::value<string>()->default_value(""), "Which models should be dropped out (zero-indexed ints in comma-delimited groups separated by spaces)")
      ("dropout_prob", po::value<float>()->default_value(0.0), "Starting dropout probability")
      ("dropout_prob_decay", po::value<float>()->default_value(1.0), "Dropout probability decay (1.0 for no decay)")
      ("heuristic", po::value<string>()->default_value(""), "Type of heuristic to use")
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

  // Create the timer
  Timer time;
  cout << "Started training! (s=" << time.Elapsed() << ")" << endl;

  // Temporary buffers
  string line;
  vector<string> strs, strs2;

  // Save various settings
  GlobalVars::verbose = vm_["verbose"].as<int>();
  word_hist_ = vm_["word_hist"].as<int>();
  word_rep_ = vm_["word_rep"].as<int>();
  use_context_ = vm_["use_context"].as<bool>();
  penalize_unk_ = vm_["penalize_unk"].as<bool>();
  epochs_ = vm_["epochs"].as<int>();
  max_minibatch_ = vm_["max_minibatch"].as<int>();
  online_epochs_ = vm_["online_epochs"].as<int>();
  trainer_ = vm_["trainer"].as<string>();
  learning_rate_ = vm_["learning_rate"].as<float>();
  clipping_enabled_ = vm_["clipping_enabled"].as<bool>();
  float rate_decay = vm_["rate_decay"].as<float>();
  string model_out_file = vm_["model_out"].as<string>();
  dropout_prob_ = vm_["dropout_prob"].as<float>();
  dropout_prob_decay_ = vm_["dropout_prob_decay"].as<float>();

  // Create a heuristic if using one
  if(vm_["heuristic"].as<string>() != "")
    heuristic_ = HeuristicFactory::create_heuristic(vm_["heuristic"].as<string>());

  // Calculate the number of layers
  boost::split(strs, vm_["layers"].as<string>(), boost::is_any_of(" "));
  vector<int> hidden_size;
  for(auto str : strs) if(str != "") hidden_size.push_back(stoi(str));
  boost::split(strs, vm_["dropout_models"].as<string>(), boost::is_any_of(" "));
  vector<set<int> > dropout_models;
  for(auto str : strs) if(str != "") {
    boost::split(strs2, str, boost::is_any_of(","));
    set<int> my_set;
    for(auto str2 : strs2) if(str2 != "") {
      my_set.insert(stoi(str2));
    }
    dropout_models.push_back(my_set);
  }
  dropout_spans_.resize(dropout_models.size());

  // Get files:
  vector<string> train_files, wildcards, test_files;
  string valid_file;
  boost::split(wildcards, vm_["wildcards"].as<string>(), boost::is_any_of(" "));
  train_files = split_wildcarded(vm_["train_file"].as<string>(), wildcards, "|", true);
  if(train_files.size() < 1 || train_files[0] == "") THROW_ERROR("Must specify at least one --train_file");
  valid_file = vm_["valid_file"].as<string>();
  boost::split(test_files, vm_["test_file"].as<string>(), boost::is_any_of("|"));
  if(test_files.size() < 1 || test_files[0] == "") THROW_ERROR("Must specify at least one --test_file");

  cout << "Reading vocabulary... (s=" << time.Elapsed() << ")" << endl;

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
    vector<string> my_locs = split_wildcarded(str, wildcards, "|", false);
    if(my_locs.size() != 1 && my_locs.size() != train_files.size() + 1)
      THROW_ERROR("When using cross-validation on the training data, must have appropriate model size.");
    model_locs.push_back(my_locs);
  }

  // Read in the the testing distributions
  vector<DistPtr> dists;
  size_t max_ctxt = word_hist_;
  for(auto & locs : model_locs) {
    cout << "Started reading model " << locs[0] << " (s=" << time.Elapsed() << ")" << endl;
    DistPtr dist = DistFactory::from_file(locs[0], dict);
    dists.push_back(dist);
    max_ctxt = max(dist->get_ctxt_len(), max_ctxt);
    num_dense_dist_ += dist->get_dense_size();
    num_sparse_dist_ += dist->get_sparse_size();
    num_ctxt_ += dist->get_ctxt_size();
  }
  cout << "Finished reading models (s=" << time.Elapsed() << ")" << endl;

  // Find the spans for dropout if necessary
  if(dropout_spans_.size() != 0) {
    size_t mod_id = 0;
    size_t curr_dense = 0, curr_sparse = 0, dense_end = 0, sparse_end = 0;
    for(size_t did = 0; did < dists.size(); did++, curr_dense = dense_end, curr_sparse = sparse_end) {
      dense_end += dists[did]->get_dense_size();
      sparse_end += dists[did]->get_sparse_size();
      for(size_t i = 0; i < dropout_spans_.size(); i++) {
        if(dropout_models[i].find(mod_id) == dropout_models[i].end()) {
          for(size_t j = curr_dense; j < dense_end; j++)
            dropout_spans_[i].push_back(j);
          for(size_t j = curr_sparse; j < sparse_end; j++)
            dropout_spans_[i].push_back(j+num_dense_dist_);
        }
      }
      mod_id++;
    }
  }

  // Create the testing/validation instances
  TrainingData train_inst, valid_inst;
  pair<int,int> train_words(0,0), valid_words(0,0);
  vector<TrainingData> test_inst(test_files.size());
  vector<pair<int,int> > test_words(test_files.size(), pair<int,int>(0,0));
  TrainingDataMap data_map;
  if(valid_file != "") {
    cout << "Creating data for " << valid_file << " (s=" << time.Elapsed() << ")" << endl;
    valid_words = create_data(dists, max_ctxt, false, dict, valid_file, data_map);
    convert_data(data_map, valid_inst);
    data_map.clear();
  }
  for(size_t i = 0; i < test_files.size(); i++) {
    cout << "Creating data for " << test_files[i] << " (s=" << time.Elapsed() << ")" << endl;
    test_words[i]  = create_data(dists, max_ctxt, false, dict, test_files[i], data_map);
    convert_data(data_map, test_inst[i]);
    data_map.clear();
  }

  // Create the training instances
  for(size_t i = 0; i < train_files.size(); i++) {
    for(size_t j = 0; j < model_locs.size(); j++) {
      if(model_locs[j].size() != 1) {
        cout << "Started reading model " << model_locs[j][i+1] << " (s=" << time.Elapsed() << ")" << endl;
        dists[j] = DistFactory::from_file(model_locs[j][i+1], dict);
      }
    }
    cout << "Creating data for " << train_files[i] << " (s=" << time.Elapsed() << ")" << endl;
    pair<int,int> my_words = create_data(dists, max_ctxt, vm_["hold_out"].as<bool>(), dict, train_files[i], data_map);
    train_words.first += my_words.first; train_words.second += my_words.second;
  }
  convert_data(data_map, train_inst);
  dists.clear();
  cout << "Done creating data. Whitening... (s=" << time.Elapsed() << ")" << endl;

  // Whiten the data if necessary
  Whitener whitener(vm_["whiten"].as<string>(), vm_["whiten_eps"].as<float>());
  whitener.calc_matrix(train_inst);
  whitener.whiten(train_inst);
  whitener.whiten(valid_inst);
  for(auto & my_inst : test_inst)
    whitener.whiten(my_inst);


  cout << "Creating model (s=" << time.Elapsed() << ")" << endl;

  // Initialize
  cnn::Model mod;
  TrainerPtr trainer = get_trainer(trainer_, learning_rate_, mod);
  trainer->clipping_enabled = clipping_enabled_;

  float uniform_prob = 1.0/dict->size();
  log_unk_prob_ = vm_["penalize_unk"].as<bool>() ? log(uniform_prob) : 0;

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
  float last_valid = 1e99, best_valid = 1e99;
  for(int epoch = 1; epoch <= epochs_; epoch++) { 
    cout << "--- Starting epoch " << epoch << " (s=" << time.Elapsed() << ")" << endl;
    calc_instance(train_inst, "trn ", train_words, true, epoch, trainer, mod);
    if(valid_inst.size() != 0) {
      float valid_loss = calc_instance(valid_inst, "vld ", valid_words, false, epoch, trainer, mod);
      if(rate_decay != 1.0 && last_valid < valid_loss) {
        trainer->eta0 *= rate_decay;
        cout << "*** Reduced learning rate to " << trainer->eta0 << endl;
      }
      last_valid = valid_loss;
      // Open the output model
      if(best_valid > valid_loss && model_out_file != "") {
        ofstream out(model_out_file.c_str());
        if(!out) THROW_ERROR("Could not open output file: " << model_out_file);
        boost::archive::text_oarchive oa(out);
        oa << mod;
        best_valid = valid_loss;
      }
    }
    for(size_t i = 0; i < test_inst.size(); i++) {
      ostringstream oss;
      calc_instance(test_inst[i], "tst" + to_string(i), test_words[i], false, epoch, trainer, mod);
    }
    // Reset the trainer after online learning
    if(epoch == online_epochs_) {
      trainer = get_trainer(trainer_, learning_rate_, mod);
      trainer->clipping_enabled = clipping_enabled_;
    }
    dropout_prob_ *= dropout_prob_decay_;
  }

  cout << "Done training! (s=" << time.Elapsed() << ")" << endl;

  return 0;
}

ModlmTrain::TrainerPtr ModlmTrain::get_trainer(const string & trainer_id, float learning_rate, cnn::Model & model) {
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

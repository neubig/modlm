#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/range/irange.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
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
#include <modlm/sequence-indexer.h>
#include <modlm/input-file-stream.h>

using namespace std;
using namespace modlm;
using namespace cnn::expr;
namespace po = boost::program_options;

// *************** Auxiliary functions

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

// *************** Calculation for both sentence-based and aggregate training

Expression ModlmTrain::add_to_graph(const std::vector<float> & ctxt_feats,
                                    const std::vector<WordId> & ctxt_words,
                                    const vector<float> & out_dists,
                                    const vector<float> & out_cnts,
                                    bool dropout,
                                    cnn::ComputationGraph & cg) {
  if(builder_.get() != NULL)
    THROW_ERROR("Recurrent nets are not supported yet");
  // Load the targets
  int num_dist = num_sparse_dist_ + num_dense_dist_;
  int num_words = out_cnts.size();
  Expression probs = input(cg, {(unsigned int)num_dist, (unsigned int)num_words}, out_dists);
  Expression counts = input(cg, {(unsigned int)num_words}, out_cnts);

  // cerr << "out_cnts:  " << print_vec(out_cnts) << endl;
  // cerr << "out_dists: " << print_vec(out_dists) << endl;
  // cerr << "ctxt_feats:  " << print_vec(ctxt_feats) << endl;

  // If not using context, it's really simple
  if(!use_context_) {
    if(dropout_prob_ > 0) THROW_ERROR("dropout not implemented for no context");
    Expression nlprob = -log(transpose(probs) * softmax( parameter(cg, a_) ) );
    Expression nll = transpose(counts) * nlprob;
    return nll;
  }

  // If using heuristics, then perform heuristic smoothing
  if(heuristic_.get() != NULL) {
    vector<float> interp = heuristic_->smooth(num_dense_dist_, ctxt_feats);
    Expression nlprob = -log( transpose(probs) * input(cg, {(unsigned int)num_dist}, interp) );
    Expression nll = transpose(counts) * nlprob;
    return nll;
  }

  // Add the context for this instance
  Expression h = input(cg, {(unsigned int)ctxt_feats.size()}, ctxt_feats);

  // Do the NN computation
  if(word_hist_ != 0) {
    vector<Expression> expr_cat;
    if(ctxt_feats.size() != 0)
      expr_cat.push_back(h);
    for(size_t i = 0; i < ctxt_words.size(); i++)
      expr_cat.push_back(lookup(cg, reps_, ctxt_words[i]));
    h = (expr_cat.size() > 1 ? concatenate(expr_cat) : expr_cat[0]);
  }
  for(size_t i = 0; i < Ws_.size(); i++)
    h = tanh( parameter(cg, Ws_[i]) * h + parameter(cg, bs_[i]) );

  Expression softmax_input = parameter(cg, V_) * h + parameter(cg, a_);
  // Calculate which interpolation coefficients to use
  int dropout_set = dropout ? calc_dropout_set() : -1;
  // Calculate the interpolation coefficients, dropping some out if necessary
  Expression interp = (dropout_set < 0 ?
                       softmax( softmax_input ) :
                       exp( log_softmax( softmax_input, dropout_spans_[dropout_set] ) ));
  Expression nlprob = -log(transpose(probs) * interp);
  Expression nll = transpose(counts) * nlprob;
  return nll;
}

int ModlmTrain::calc_dropout_set() {
  uniform_real_distribution<float> float_distribution(0.0, 1.0);  
  if(float_distribution(*cnn::rndeng) >= dropout_prob_) {
    return -1;
  } else {
    assert(dropout_spans_.size() > 0);
    uniform_int_distribution<int> int_distribution(0, (int)dropout_spans_.size()-1);  
    return int_distribution(*cnn::rndeng);
  }
}

// *************** Calculation for aggregate training

Expression ModlmTrain::create_aggregate_graph(const IndexedAggregateInstance & inst, pair<size_t,size_t> range, pair<int,int> & words, bool dropout, cnn::ComputationGraph & cg) {
  // Dynamically create the target vectors
  int num_dist = num_dense_dist_ + num_sparse_dist_;
  int num_words = (range.second - range.first);
  vector<float> wdists_(num_words * num_dist, 0.0), wcnts(num_words);
  size_t ptr = 0;
  for(size_t pos = range.first; pos < range.second; pos++) {
    auto & kv = inst.second[pos];
    memcpy(&wdists_[ptr], &dist_inverter_[kv.first.first][0], sizeof(float)*num_dense_dist_);
    ptr += num_dense_dist_;
    for(auto & elem : kv.first.second)
      wdists_[ptr + elem.first] = elem.second;
    ptr += num_sparse_dist_;
    wcnts[pos-range.first] = kv.second;
    words.first += kv.second;
  }
  return add_to_graph(ctxt_inverter_[inst.first.first], inst.first.second, wdists_, wcnts, dropout, cg);
}

float ModlmTrain::calc_aggregate_instance(const IndexedAggregateData & data, const std::string & strid, std::pair<int,int> words, bool update, int epoch) {
  float loss = 0.0, print_every = 60.0, elapsed;
  float last_print = 0;
  Timer time;
  pair<int,int> curr_words(0,0);
  int data_done = 0;
  for(auto inst : data) {
    for(size_t i = 0; i < inst.second.size(); i += max_minibatch_) {
      cnn::ComputationGraph cg;
      create_aggregate_graph(inst, make_pair(i, min(inst.second.size(), i+max_minibatch_)), curr_words, update, cg);
      loss += cnn::as_scalar(cg.forward());
      if(loss != loss) THROW_ERROR("Loss is not a number");
      if(update) {
        cg.backward();
        if(online_epochs_ == -1 || epoch <= online_epochs_)
          trainer_->update();
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
      trainer_->update();
    trainer_->update_epoch();
  }
  return loss;
}


void ModlmTrain::convert_aggregate_data(const IndexedAggregateDataMap & data_map, IndexedAggregateData & data) {
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

void ModlmTrain::sanity_check_aggregate(const SequenceIndexer<Sentence> & my_counts, float uniform_prob, float unk_prob) {
  cerr << "Performing sanity check" << endl;
  std::unordered_set<Sentence> checked_ctxts;
  for(const auto & my_count : my_counts.get_index()) {
    Sentence my_ctxt(my_count.first); my_ctxt.resize(my_ctxt.size()-1);
    if(checked_ctxts.find(my_ctxt) == checked_ctxts.end()) {
      Sentence my_ngram(my_count.first);
      vector<float> dist_trg_sum(num_dense_dist_);
      for(WordId wid = 0; wid < dict_->size(); wid++) {
        *my_ngram.rbegin() = wid;
        std::vector<float> trg_dense(num_dense_dist_);
        std::vector<std::pair<int, float> > trg_sparse;
        int dense_offset = 0, sparse_offset = 0;
        for(auto dist : dists_)
          dist->calc_word_dists(my_ngram, uniform_prob, unk_prob, trg_dense, dense_offset, trg_sparse, sparse_offset);
        // cerr << "   " << dict_->Convert(wid) << ":";
        for(size_t did = 0; did < num_dense_dist_; did++) {
          dist_trg_sum[did] += trg_dense[did] / (wid == 0 ? unk_prob : 1.f);
          // cerr << ' ' << dist_trg.first[did];
        }
        // cerr << endl;
      }
      for(size_t did = 0; did < num_dense_dist_; did++) {
        // cerr << "Distribution " << did << " for " << PrintSentence(my_ctxt, dict_) << ": " << dist_trg_sum[did] << endl;
        if(dist_trg_sum[did] > 1.01f || dist_trg_sum[did] < 0.99f)
          THROW_ERROR("Distribution " << did << " for " << PrintSentence(my_ctxt, dict_) << " > 1: " << dist_trg_sum[did]);
      }
      checked_ctxts.insert(my_ctxt);
    }
  }
  cerr << "Sanity check passed" << endl;
}

pair<int,int> ModlmTrain::create_aggregate_data(const string & file_name, IndexedAggregateDataMap & data) {

  float uniform_prob = 1.0/dict_->size();
  float unk_prob = (penalize_unk_ ? uniform_prob : 1);
  
  // Calculate n-gram counts with a profile
  SequenceIndexer<Sentence> my_counts(max_ctxt_+1);

  // Load counts
  {
    ifstream in(file_name);
    if(!in) THROW_ERROR("Could not open in create_instances: " << file_name);
    string line;
    while(getline(in, line)) {
      Sentence sent = ParseSentence(line, dict_, true);
      my_counts.add_counts(sent);
    }
  }

  // // Perform a sanity check
  // sanity_check_aggregate(my_counts, uniform_prob, unk_prob);

  // Create training data (num words, ctxt features, each model, true counts)
  pair<int,int> total_words(0,0);
  // Create and allocate the targets:
  std::vector<float> ctxt_dense(num_ctxt_);
  std::vector<WordId> ctxt_sparse(word_hist_);
  // Loop through each of the contexts
  for(auto & kv : my_counts.get_index()) {
    // Create a vector containing only the context words
    Sentence ctxt_words = kv.first;
    ctxt_words.resize(ctxt_words.size() - 1);
    // Calculate the words and contexts
    for(int i = 0; i < word_hist_; i++)
      ctxt_sparse[i] = ctxt_words[ctxt_words.size()-word_hist_+i];
    int curr_ctxt = 0;
    for(auto dist : dists_) {
      assert(dist.get() != NULL);
      dist->calc_ctxt_feats(ctxt_words, &ctxt_dense[curr_ctxt]);
      curr_ctxt += dist->get_ctxt_size();
    }
    // Prepare the pointers for each word
    std::vector<float> trg_dense(num_dense_dist_);
    std::vector<std::pair<int,float> > trg_sparse;
    total_words.first += kv.second;
    if(*kv.first.rbegin() == 0) total_words.second += kv.second;
    // Calculate all of the distributions
    int dense_offset = 0, sparse_offset = 0;
    for(auto dist : dists_)
      dist->calc_word_dists(kv.first, uniform_prob, unk_prob, trg_dense, dense_offset, trg_sparse, sparse_offset);
    // Add counts for the context
    IndexedAggregateContext full_ctxt(ctxt_indexer_.get_index(ctxt_dense, true), ctxt_sparse);
    IndexedDistTarget full_trg(dist_indexer_.get_index(trg_dense, true), trg_sparse);
    data[full_ctxt][full_trg] += kv.second;
  } 

  return total_words;
}

void ModlmTrain::train_aggregate() {

  // Create the testing/validation instances
  IndexedAggregateData train_inst, valid_inst;
  pair<int,int> train_words(0,0), valid_words(0,0);
  vector<IndexedAggregateData> test_inst(test_files_.size());
  vector<pair<int,int> > test_words(test_files_.size(), pair<int,int>(0,0));
  IndexedAggregateDataMap data_map;
  if(valid_file_ != "") {
    cout << "Creating data for " << valid_file_ << " (s=" << time_.Elapsed() << ")" << endl;
    valid_words = create_aggregate_data(valid_file_, data_map);
    convert_aggregate_data(data_map, valid_inst);
    data_map.clear();
  }
  for(size_t i = 0; i < test_files_.size(); i++) {
    cout << "Creating data for " << test_files_[i] << " (s=" << time_.Elapsed() << ")" << endl;
    test_words[i]  = create_aggregate_data(test_files_[i], data_map);
    convert_aggregate_data(data_map, test_inst[i]);
    data_map.clear();
  }

  // Create the training instances
  for(size_t i = 0; i < train_files_.size(); i++) {
    for(size_t j = 0; j < model_locs_.size(); j++) {
      if(model_locs_[j].size() != 1) {
        cout << "Started reading model " << model_locs_[j][i+1] << " (s=" << time_.Elapsed() << ")" << endl;
        dists_[j] = DistFactory::from_file(model_locs_[j][i+1], dict_);
      }
    }
    cout << "Creating data for " << train_files_[i] << " (s=" << time_.Elapsed() << ")" << endl;
    pair<int,int> my_words = create_aggregate_data(train_files_[i], data_map);
    train_words.first += my_words.first; train_words.second += my_words.second;
  }
  convert_aggregate_data(data_map, train_inst);
  dists_.clear();
  cout << "Done creating data. Whitening... (s=" << time_.Elapsed() << ")" << endl;

  // Now that we're done creating indexed data, invert the index
  dist_indexer_.build_inverse_index(dist_inverter_); dist_indexer_.get_index().clear();
  ctxt_indexer_.build_inverse_index(ctxt_inverter_); ctxt_indexer_.get_index().clear();

  // Whiten the data if necessary
  if(whitener_.get() != NULL) {
    THROW_ERROR("Whitening not re-implemented yet");
    // whitener_->calc_matrix(train_inst);
    // whitener_->whiten(train_inst);
    // whitener_->whiten(valid_inst);
    // for(auto & my_inst : test_inst)
    //   whitener_->whiten(my_inst);
  }

  // Train a neural network to predict_ the interpolation coefficients
  float last_valid = 1e99, best_valid = 1e99;
  for(int epoch = 1; epoch <= epochs_; epoch++) { 
    // Print info about the epoch
    cout << "--- Starting epoch " << epoch << ": "<<(epoch<=online_epochs_?"online":"batch")<<", lr=" << trainer_->eta0;
    if(dropout_prob_ != 0.0)
      cout << ", dropout=" << min(dropout_prob_, 1.0f);
    cout << " (s=" << time_.Elapsed() << ")" << endl;
    // Perform training
    calc_aggregate_instance(train_inst, "trn ", train_words, true, epoch);
    if(valid_inst.size() != 0) {
      float valid_loss = calc_aggregate_instance(valid_inst, "vld ", valid_words, false, epoch);
      if(rate_decay_ != 1.0 && last_valid < valid_loss)
        trainer_->eta0 *= rate_decay_;
      last_valid = valid_loss;
      // Open the output model
      if(best_valid > valid_loss && model_out_file_ != "") {
        ofstream out(model_out_file_.c_str());
        if(!out) THROW_ERROR("Could not open output file: " << model_out_file_);
        boost::archive::text_oarchive oa(out);
        oa << *mod_;
        best_valid = valid_loss;
      }
    }
    for(size_t i = 0; i < test_inst.size(); i++) {
      ostringstream oss;
      calc_aggregate_instance(test_inst[i], "tst" + to_string(i), test_words[i], false, epoch);
    }
    // Reset the trainer after online learning
    if(epoch == online_epochs_) {
      trainer_ = get_trainer(trainer_id_, learning_rate_, *mod_);
      trainer_->clipping_enabled = clipping_enabled_;
    }
    dropout_prob_ *= dropout_prob_decay_;
  }

  cout << "Done training! (s=" << time_.Elapsed() << ")" << endl;

}

// *************** Calculation for both sentence-based and aggregate training

void ModlmTrain::train_sentencewise() {
  THROW_ERROR("Not finished yet");
}

// *************** Calculation for both sentence-based and aggregate training

int ModlmTrain::main(int argc, char** argv) {
  po::options_description desc("*** modlm-train (by Graham Neubig) ***");
  desc.add_options()
      ("help", "Produce help message")
      ("clipping_enabled", po::value<bool>()->default_value(true), "Whether to enable clipping or not")
      ("cnn_mem", po::value<int>()->default_value(512), "Memory used by cnn in megabytes")
      ("cnn_seed", po::value<int>()->default_value(0), "Random seed (default 0 -> changes every time)")
      ("dist_models", po::value<string>()->default_value(""), "Files containing the distribution models")
      ("dropout_models", po::value<string>()->default_value(""), "Which models should be dropped out (zero-indexed ints in comma-delimited groups separated by spaces)")
      ("dropout_prob", po::value<float>()->default_value(0.0), "Starting dropout probability")
      ("dropout_prob_decay", po::value<float>()->default_value(1.0), "Dropout probability decay (1.0 for no decay)")
      ("epochs", po::value<int>()->default_value(300), "Number of epochs")
      ("heuristic", po::value<string>()->default_value(""), "Type of heuristic to use")
      ("layers", po::value<string>()->default_value("50"), "Descriptor for hidden layers, e.g. 50_30")
      ("learning_rate", po::value<float>()->default_value(0.1), "Learning rate")
      ("max_minibatch", po::value<int>()->default_value(256), "Max minibatch size")
      ("model_in", po::value<string>()->default_value(""), "If resuming training, read the model in")
      ("model_out", po::value<string>()->default_value(""), "File to write the model to")
      ("online_epochs", po::value<int>()->default_value(-1), "Number of epochs of online learning to perform before switching to batch (-1: only online)")
      ("penalize_unk", po::value<bool>()->default_value(true), "Whether to penalize unknown words")
      ("rate_decay", po::value<float>()->default_value(1.0), "How much to decay learning rate when validation likelihood decreases")
      ("rate_thresh",  po::value<float>()->default_value(1e-5), "Threshold for the learning rate")
      ("test_file", po::value<string>()->default_value(""), "One or more testing files split with pipes")
      ("train_file", po::value<string>()->default_value(""), "One or more training files split with pipes")
      ("trainer", po::value<string>()->default_value("adam"), "Training algorithm (sgd/momentum/adagrad/adadelta/adam)")
      ("training_type", po::value<string>()->default_value("agg"), "Train using aggregate or sentence-wise examples (agg/sent)")
      ("use_context", po::value<bool>()->default_value(true), "If set to false, learn context-independent coefficients")
      ("valid_file", po::value<string>()->default_value(""), "Validation file for tuning parameters")
      ("verbose", po::value<int>()->default_value(0), "How much verbose output to print")
      ("vocab_file", po::value<string>()->default_value(""), "Vocab file")
      ("whiten", po::value<string>()->default_value(""), "Type of whitening (mean/pca/zca)")
      ("whiten_eps", po::value<float>()->default_value(0.01), "Regularization for whitening")
      ("wildcards", po::value<string>()->default_value(""), "Wildcards in model/data names for cross validation")
      ("word_hist", po::value<int>()->default_value(0), "Word history length")
      ("word_rep", po::value<int>()->default_value(50), "Word representation size")
      ;
  boost::program_options::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);   
  if (vm.count("help")) {
      cout << desc << endl;
      return 1;
  }

  // Create the timer
  time_ = Timer();
  cout << "Started training! (s=" << time_.Elapsed() << ")" << endl;

  // Temporary buffers
  string line;
  vector<string> strs, strs2;

  // Save various settings
  GlobalVars::verbose = vm["verbose"].as<int>();
  word_hist_ = vm["word_hist"].as<int>();
  word_rep_ = vm["word_rep"].as<int>();
  use_context_ = vm["use_context"].as<bool>();
  penalize_unk_ = vm["penalize_unk"].as<bool>();
  epochs_ = vm["epochs"].as<int>();
  max_minibatch_ = vm["max_minibatch"].as<int>();
  online_epochs_ = vm["online_epochs"].as<int>();
  trainer_id_ = vm["trainer"].as<string>();
  learning_rate_ = vm["learning_rate"].as<float>();
  clipping_enabled_ = vm["clipping_enabled"].as<bool>();
  training_type_ = vm["training_type"].as<string>();
  rate_decay_ = vm["rate_decay"].as<float>();
  model_out_file_ = vm["model_out"].as<string>();
  dropout_prob_ = vm["dropout_prob"].as<float>();
  dropout_prob_decay_ = vm["dropout_prob_decay"].as<float>();

  // Create a heuristic if using one
  if(vm["heuristic"].as<string>() != "")
    heuristic_ = HeuristicFactory::create_heuristic(vm["heuristic"].as<string>());
  if(vm["whiten"].as<string>() != "")
    whitener_.reset(new Whitener(vm["whiten"].as<string>(), vm["whiten_eps"].as<float>()));

  // Calculate the number of layers
  boost::split(strs, vm["layers"].as<string>(), boost::is_any_of(" "));
  vector<int> hidden_size;
  for(auto str : strs) if(str != "") hidden_size.push_back(stoi(str));
  boost::split(strs, vm["dropout_models"].as<string>(), boost::is_any_of(" "));
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
  vector<string> wildcards;
  boost::split(wildcards, vm["wildcards"].as<string>(), boost::is_any_of(" "));
  train_files_ = split_wildcarded(vm["train_file"].as<string>(), wildcards, "|", true);
  if(train_files_.size() < 1 || train_files_[0] == "") THROW_ERROR("Must specify at least one --train_file");
  valid_file_ = vm["valid_file"].as<string>();
  boost::split(test_files_, vm["test_file"].as<string>(), boost::is_any_of("|"));
  if(test_files_.size() < 1 || test_files_[0] == "") THROW_ERROR("Must specify at least one --test_file");

  cout << "Reading vocabulary... (s=" << time_.Elapsed() << ")" << endl;

  // Read in the vocabulary if necessary
  dict_.reset(new cnn::Dict);
  dict_->Convert("<unk>");
  dict_->Convert("<s>");
  string vocab_file = vm["vocab_file"].as<string>();
  if(vocab_file == "")
    THROW_ERROR("Must specify a vocabulary file");
  InputFileStream vocab_in(vocab_file);
  if(!(getline(vocab_in, line) && line == "<unk>" && getline(vocab_in, line) && line == "<s>"))
    THROW_ERROR("First two lines of a vocabulary file must be <unk> and <s>: " << vocab_file);
  while(getline(vocab_in, line))
    dict_->Convert(line);
  dict_->Freeze();
  dict_->SetUnk("<unk>");

  // Read in the model locations. For each type of model, there must be either one model, or one
  // testing model, and then a model for each of the training folds
  vector<string> model_types;
  boost::split(model_types, vm["dist_models"].as<string>(), boost::is_any_of(" "));
  for(auto str : model_types) {
    vector<string> my_locs = split_wildcarded(str, wildcards, "|", false);
    if(my_locs.size() != 1 && my_locs.size() != train_files_.size() + 1)
      THROW_ERROR("When using cross-validation on the training data, must have appropriate model size.");
    model_locs_.push_back(my_locs);
  }

  // Read in the the testing distributions
  max_ctxt_ = word_hist_;
  for(auto & locs : model_locs_) {
    cout << "Started reading model " << locs[0] << " (s=" << time_.Elapsed() << ")" << endl;
    DistPtr dist = DistFactory::from_file(locs[0], dict_);
    dists_.push_back(dist);
    max_ctxt_ = max((int)dist->get_ctxt_len(), max_ctxt_);
    num_dense_dist_ += dist->get_dense_size();
    num_sparse_dist_ += dist->get_sparse_size();
    num_ctxt_ += dist->get_ctxt_size();
  }
  cout << "Finished reading models (s=" << time_.Elapsed() << ")" << endl;

  // Find the spans for dropout if necessary
  if(dropout_spans_.size() != 0) {
    size_t mod_id = 0;
    size_t curr_dense = 0, curr_sparse = 0, dense_end = 0, sparse_end = 0;
    for(size_t did = 0; did < dists_.size(); did++, curr_dense = dense_end, curr_sparse = sparse_end) {
      dense_end += dists_[did]->get_dense_size();
      sparse_end += dists_[did]->get_sparse_size();
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

  cout << "Creating model (s=" << time_.Elapsed() << ")" << endl;

  // Initialize
  mod_.reset(new cnn::Model);
  trainer_ = get_trainer(trainer_id_, learning_rate_, *mod_);
  trainer_->clipping_enabled = clipping_enabled_;

  float uniform_prob = 1.0/dict_->size();
  log_unk_prob_ = vm["penalize_unk"].as<bool>() ? log(uniform_prob) : 0;

  int num_dist = num_sparse_dist_ + num_dense_dist_;
  if(use_context_) {
    int last_size = num_ctxt_ + word_rep_ * word_hist_;
    // Add the word representation and transformation functions
    if(word_hist_ != 0)
      reps_ = mod_->add_lookup_parameters(dict_->size(), {(unsigned int)word_rep_});
    // Add the functions
    for(auto size : hidden_size) {
      Ws_.push_back(mod_->add_parameters({(unsigned int)size, (unsigned int)last_size}));
      bs_.push_back(mod_->add_parameters({(unsigned int)size}));
      last_size = size;
    }
    V_ = mod_->add_parameters({(unsigned int)num_dist, (unsigned int)last_size});
  }
  a_ = mod_->add_parameters({(unsigned int)num_dist});

  // Actually perform training
  if(training_type_ == "agg") {
    train_aggregate();
  } else if(training_type_ == "sent") {
    train_sentencewise();
  } else {
    THROW_ERROR("Illegal training type");
  }

  return 0;

}

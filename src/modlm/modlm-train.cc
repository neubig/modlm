#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <numeric>
#include <boost/program_options.hpp>
#include <boost/range/irange.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <cnn/expr.h>
#include <cnn/cnn.h>
#include <cnn/dict.h>
#include <cnn/training.h>
#include <cnn/rnn.h>
#include <cnn/lstm.h>
#include <cnn/grad-check.h>
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
using namespace cnn::expr;
namespace po = boost::program_options;

namespace modlm {

// *************** Auxiliary functions

template <class T>
inline std::string print_vec(const std::vector<T> vec) {
  ostringstream oss;
  if(vec.size()) oss << vec[0];
  for(int i : boost::irange(1, (int)vec.size()))
    oss << ' ' << vec[i];
  return oss.str();
}

void ModlmTrain::print_status(const std::string & strid, std::pair<int,int> epoch, float loss, pair<int,int> words, float percent, float elapsed) {
  float ppl = exp(loss/words.first);
  float ppl_nounk = exp((loss + words.second * log_unk_prob_)/words.first);
  float wps = words.first / elapsed;
  cout << strid << " epoch " << epoch.first;
  if(epoch.second != 0) cout << "-" << epoch.second;
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

ModlmTrain::TrainerPtr ModlmTrain::get_trainer(const string & trainer_id, float learning_rate, float weight_decay, cnn::Model & model) {
    TrainerPtr trainer;
    if(trainer_id == "sgd") {
        trainer.reset(new cnn::SimpleSGDTrainer(&model, weight_decay, learning_rate));
    } else if(trainer_id == "momentum") {
        trainer.reset(new cnn::MomentumSGDTrainer(&model, weight_decay, learning_rate));
    } else if(trainer_id == "adagrad") {
        trainer.reset(new cnn::AdagradTrainer(&model, weight_decay, learning_rate));
    } else if(trainer_id == "adadelta") {
        trainer.reset(new cnn::AdadeltaTrainer(&model, weight_decay, learning_rate));
    } else if(trainer_id == "adam") {
        trainer.reset(new cnn::AdamTrainer(&model, weight_decay, learning_rate));
    } else if(trainer_id == "rms") {
        trainer.reset(new cnn::RmsPropTrainer(&model, weight_decay, learning_rate));
    } else {
        THROW_ERROR("Illegal trainer variety: " << trainer_id);
    }
    return trainer;
}

// ****** Create the graph

Expression ModlmTrain::add_to_graph(const std::vector<float> & ctxt_feats,
                                    const std::vector<WordId> & ctxt_words,
                                    const vector<float> & out_dists,
                                    const vector<float> & out_cnts,
                                    bool dropout,
                                    cnn::ComputationGraph & cg) {
  // Load the targets
  int num_dist = num_sparse_dist_ + num_dense_dist_;
  int num_words = out_cnts.size();
  Expression interp;

  // cerr << "out_cnts:  " << print_vec(out_cnts) << endl;
  // cerr << "out_dists: " << print_vec(out_dists) << endl;
  // cerr << "ctxt_feats:  " << print_vec(ctxt_feats) << endl;
  // cerr << "a: " << print_vec(as_vector(a_expr_.value())) << endl;

  // If not using context, just use the bias
  if(!use_context_) {
    if(model_dropout_prob_ > 0) THROW_ERROR("dropout not implemented for no context");
    interp = softmax( a_expr_ );
  // If using heuristics, then perform heuristic smoothing
  } else if (heuristic_.get() != NULL) {
    vector<float> interp_vals = heuristic_->smooth(num_dense_dist_, ctxt_feats);
    interp = input(cg, {(unsigned int)num_dist}, interp_vals);
  // Otherwise, computer using a neural net
  } else {

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

    if(builder_.get() != NULL)
      h = builder_->add_input(h);

    Expression softmax_input = affine_transform({a_expr_, V_expr_, h});
    // Calculate which interpolation coefficients to use
    int dropout_set = dropout ? calc_dropout_set() : -1;
    // Calculate the interpolation coefficients, dropping some out if necessary
    if(dropout_set < 0)
      interp = softmax( softmax_input );
    else 
      interp = exp( log_softmax( softmax_input, dropout_spans_[dropout_set] ) );

  }

  Expression probs = input(cg, {(unsigned int)num_dist, (unsigned int)num_words}, out_dists);
  Expression nll = -log(transpose(probs) * interp);  
  if(out_cnts.size() > 1 || out_cnts[0] > 1) {
    Expression counts = input(cg, {(unsigned int)num_words}, out_cnts);
    nll = transpose(counts) * nll;
  }

  return nll;
}

template <>
float ModlmTrain::calc_instance<IndexedAggregateInstance>(const IndexedAggregateInstance & inst, bool update, std::pair<int,int> epoch, pair<int,int> & words) {
  int num_dist = num_dense_dist_ + num_sparse_dist_;
  float loss = 0.f;
  for(size_t i = 0; i < inst.second.size(); i += max_minibatch_) {
    cnn::ComputationGraph cg;
    V_expr_ = parameter(cg, V_);
    a_expr_ = parameter(cg, a_);
    if(builder_.get() != NULL) {
      builder_->new_graph(cg);
      builder_->start_new_sequence();
    }
    // Dynamically create the target vectors
    pair<int,int> range(i, min(inst.second.size(), i+max_minibatch_));
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
    add_to_graph(ctxt_inverter_[inst.first.first], inst.first.second, wdists_, wcnts, update, cg);
    loss += cnn::as_scalar(cg.forward());
    if(loss != loss) THROW_ERROR("Loss is not a number");
    if(update) {
      cg.backward();
      if(online_epochs_ == -1 || epoch.first <= online_epochs_)
        trainer_->update();
    }
  }
  return loss;
}

template <>
float ModlmTrain::calc_instance<IndexedSentenceInstance>(const IndexedSentenceInstance & inst, bool update, std::pair<int,int> epoch, pair<int,int> & words) {
  int num_dist = num_dense_dist_ + num_sparse_dist_;
  cnn::ComputationGraph cg;
  V_expr_ = parameter(cg, V_);
  a_expr_ = parameter(cg, a_);
  if(builder_.get() != NULL) {
    builder_->new_graph(cg);
    builder_->start_new_sequence();
  }
  vector<float> wcnts(1, 1.f);
  Sentence ctxt_ngram(word_hist_, 1);
  std::vector<Expression> loss_exps;
  words.first += inst.first.size();
  for(size_t i = 0; i < inst.first.size(); i++) {
    if(inst.first[i] == 0) words.second++;
    // Dynamically create the target vectors
    vector<float> wdists(num_dist, 0.0);
    auto & dist_trg = inst.second[i].second;
    // Sanity checks for memcpy
    memcpy(&wdists[0], &dist_inverter_[dist_trg.first][0], sizeof(float)*num_dense_dist_);
    for(auto & elem : dist_trg.second) {
      assert(elem.first >= 0);
      wdists[num_dense_dist_ + elem.first] = elem.second;
    }
    loss_exps.push_back(add_to_graph(ctxt_inverter_[inst.second[i].first], ctxt_ngram, wdists, wcnts, update, cg));
    if(word_hist_) { ctxt_ngram.erase(ctxt_ngram.begin()); ctxt_ngram.push_back(inst.first[i]); }
  }
  // Sum the losses and perform computation
  sum(loss_exps);
  float loss = cnn::as_scalar(cg.forward());
  if(loss != loss) THROW_ERROR("Loss is not a number");
  if(update) {
    cg.backward();
    if(online_epochs_ == -1 || epoch.first <= online_epochs_)
      trainer_->update();
  }

  return loss;
}

// ****** Calculate things for a single data set

int ModlmTrain::calc_dropout_set() {
  uniform_real_distribution<float> float_distribution(0.0, 1.0);  
  if(float_distribution(*cnn::rndeng) >= model_dropout_prob_) {
    return -1;
  } else {
    assert(dropout_spans_.size() > 0);
    uniform_int_distribution<int> int_distribution(0, (int)dropout_spans_.size()-1);  
    return int_distribution(*cnn::rndeng);
  }
}

template <class Data, class Instance>
float ModlmTrain::calc_dataset(const Data & data, const std::string & strid, std::pair<int,int> words, bool update, std::pair<int,int> epoch) {
  std::vector<int> idxs(data.size());
  std::iota(idxs.begin(), idxs.end(), 0);
  return calc_dataset<Data,Instance>(data, strid, words, update, epoch, idxs, make_pair(0, idxs.size()));
}

template <class Data, class Instance>
float ModlmTrain::calc_dataset(const Data & data, const std::string & strid, std::pair<int,int> words, bool update, std::pair<int,int> epoch, std::vector<int> & idxs, pair<size_t,size_t> range) {

  // Set the dropout for the LSTM
  if(hidden_spec_.type != "lstm")
    ((cnn::LSTMBuilder*)builder_.get())->set_dropout(update ? lstm_dropout_prob_ : 0.f);

  float loss = 0.0, print_every = 60.0, elapsed;
  float last_print = 0;
  Timer time;
  pair<int,int> curr_words(0,0);
  for(size_t my_pos = range.first; my_pos < range.second; my_pos++) {
    int id = idxs[my_pos];
    const auto & inst = data[id];
    loss += calc_instance(inst, update, epoch, curr_words);
    elapsed = time.Elapsed();
    if(elapsed > last_print + print_every) {
      print_status(strid, epoch, loss, curr_words, 100.0*my_pos/(float)idxs.size(), elapsed);
      last_print += print_every;
    }
  }
  elapsed = time.Elapsed();
  print_status(strid, epoch, loss, curr_words, -1, elapsed);
  if(update) {
    if(range.second == data.size()) {
      if(online_epochs_ != -1 && epoch.first > online_epochs_)
        trainer_->update();
      trainer_->update_epoch();
    }
  }
  return loss;
}

// ********* Create data

template <>
void ModlmTrain::finalize_data<IndexedAggregateDataMap,IndexedAggregateData>(const IndexedAggregateDataMap & data_map, IndexedAggregateData & data) {
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

template <>
pair<int,int> ModlmTrain::create_data<IndexedAggregateDataMap,IndexedAggregateData>(const string & file_name, IndexedAggregateDataMap & data, IndexedAggregateData &) {

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
  std::vector<float> ctxt_dense(num_ctxt_), trg_dense(num_dense_dist_);
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
    // Calculate all of the distributions
    std::vector<std::pair<int,float> > trg_sparse;
    int dense_offset = 0, sparse_offset = 0;
    for(auto dist : dists_)
      dist->calc_word_dists(kv.first, uniform_prob, unk_prob, trg_dense, dense_offset, trg_sparse, sparse_offset);
    // Add counts for the context
    IndexedAggregateContext full_ctxt(ctxt_indexer_.get_index(ctxt_dense, true), ctxt_sparse);
    IndexedDistTarget full_trg(dist_indexer_.get_index(trg_dense, true), trg_sparse);
    data[full_ctxt][full_trg] += kv.second;
    total_words.first += kv.second;
    if(*kv.first.rbegin() == 0) total_words.second += kv.second;
  } 

  return total_words;
}

template <>
void ModlmTrain::finalize_data<int,IndexedSentenceData>(const int & data_map, IndexedSentenceData & data) { }

template <>
pair<int,int> ModlmTrain::create_data<int,IndexedSentenceData>(const string & file_name, int & data_map, IndexedSentenceData & data) {
  float uniform_prob = 1.0/dict_->size();
  float unk_prob = (penalize_unk_ ? uniform_prob : 1);

  // Create training data (num words, ctxt features, each model, true counts)
  pair<int,int> total_words(0,0);

  // Create and allocate the targets:
  std::vector<float> ctxt_dense(num_ctxt_), trg_dense(num_dense_dist_);

  // Load counts
  ifstream in(file_name);
  if(!in) THROW_ERROR("Could not open in create_instances: " << file_name);
  string line;
  while(getline(in, line)) {
    // The things we'll need to return
    Sentence sent = ParseSentence(line, dict_, true);
    std::vector<std::pair<int, IndexedDistTarget> > ctxt_dists;
    total_words.first += sent.size();
    // Start with empty context at the beginning of the sentence
    Sentence ngram(max_ctxt_, 1);
    // For each word in the sentence
    for(size_t i = 0; i < sent.size(); i++) {
      if(sent[i] == 0) total_words.second++;
      // Calculate the dense context features
      int curr_ctxt = 0;
      for(auto dist : dists_) {
        assert(dist.get() != NULL);
        dist->calc_ctxt_feats(ngram, &ctxt_dense[curr_ctxt]);
        curr_ctxt += dist->get_ctxt_size();
      }
      // Change to the n-gram and calculate the distribution for it
      ngram.push_back(sent[i]); 
      std::vector<std::pair<int,float> > trg_sparse;
      int dense_offset = 0, sparse_offset = 0;
      for(auto dist : dists_)
        dist->calc_word_dists(ngram, uniform_prob, unk_prob, trg_dense, dense_offset, trg_sparse, sparse_offset);
      // Add the features
      IndexedDistTarget dist_trg(dist_indexer_.get_index(trg_dense, true), trg_sparse);
      ctxt_dists.push_back(make_pair(ctxt_indexer_.get_index(ctxt_dense, true), dist_trg));
      // Reduce the last word in the context
      ngram.erase(ngram.begin());
    }
    data.push_back(make_pair(sent, ctxt_dists));
  }

  return total_words;
}

// ***************** Training functions

template <class DataMap, class Data, class Instance>
void ModlmTrain::perform_training() {

  // Create the testing/validation instances
  Data train_inst, valid_inst;
  pair<int,int> train_words(0,0), valid_words(0,0);
  vector<Data> test_inst(test_files_.size());
  vector<pair<int,int> > test_words(test_files_.size(), pair<int,int>(0,0));
  if(valid_file_ != "") {
    DataMap data_map;
    cout << "Creating data for " << valid_file_ << " (s=" << time_.Elapsed() << ")" << endl;
    valid_words = create_data<DataMap,Data>(valid_file_, data_map, valid_inst);
    finalize_data<DataMap,Data>(data_map, valid_inst);
  }
  for(size_t i = 0; i < test_files_.size(); i++) {
    DataMap data_map;
    cout << "Creating data for " << test_files_[i] << " (s=" << time_.Elapsed() << ")" << endl;
    test_words[i]  = create_data<DataMap,Data>(test_files_[i], data_map, test_inst[i]);
    finalize_data<DataMap,Data>(data_map, test_inst[i]);
  }

  // Create the training instances
  {
    DataMap data_map;
    for(size_t i = 0; i < train_files_.size(); i++) {
      for(size_t j = 0; j < model_locs_.size(); j++) {
        if(model_locs_[j].size() != 1) {
          cout << "Started reading model " << model_locs_[j][i+1] << " (s=" << time_.Elapsed() << ")" << endl;
          dists_[j] = DistFactory::from_file(model_locs_[j][i+1], dict_);
        }
      }
      cout << "Creating data for " << train_files_[i] << " (s=" << time_.Elapsed() << ")" << endl;
      pair<int,int> my_words = create_data<DataMap,Data>(train_files_[i], data_map, train_inst);
      train_words.first += my_words.first; train_words.second += my_words.second;
    }
    finalize_data<DataMap,Data>(data_map, train_inst);
  }
  dists_.clear();
  cout << "Done creating data. Whitening... (s=" << time_.Elapsed() << ")" << endl;

  // Now that we're done creating indexed data, invert the index
  dist_indexer_.build_inverse_index(dist_inverter_); dist_indexer_.get_index().clear();
  ctxt_indexer_.build_inverse_index(ctxt_inverter_); ctxt_indexer_.get_index().clear();

  std::vector<int> train_idxs(train_inst.size());
  std::iota(train_idxs.begin(), train_idxs.end(), 0);
  std::vector<pair<size_t,size_t> > evaluate_ranges;
  for(int i = 0; i < evaluate_frequency_; i++)
    evaluate_ranges.push_back(make_pair(train_idxs.size()*i/evaluate_frequency_, train_idxs.size()*(i+1)/evaluate_frequency_));

  // Whiten the data if necessary
  if(whitener_.get() != NULL)
    THROW_ERROR("Whitening not re-implemented yet");

  // Train a neural network to predict_ the interpolation coefficients
  float last_valid = 1e99, best_valid = 1e99;
  for(int epoch = 1; epoch <= epochs_; ) { 
    for(size_t range = 1; range <= evaluate_frequency_; range++) {
      pair<int,int> epoch_pair(epoch, evaluate_frequency_ == 1 ? 0 : range);
      // Print info about the epoch
      cout << "--- Starting epoch " << epoch << "-" << range << ": "<<(online_epochs_==-1||epoch<=online_epochs_?"online":"batch")<<", lr=" << trainer_->eta0;
      if(model_dropout_prob_ != 0.0)
        cout << ", dropout=" << min(model_dropout_prob_, 1.0f);
      cout << " (s=" << time_.Elapsed() << ")" << endl;
      // Perform training
      calc_dataset<Data,Instance>(train_inst, "trn ", train_words, true, epoch_pair, train_idxs, evaluate_ranges[range]);
      if(valid_inst.size() != 0) {
        float valid_loss = calc_dataset<Data,Instance>(valid_inst, "vld ", valid_words, false, epoch_pair);
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
      for(size_t i = 0; i < test_inst.size(); i++)
        calc_dataset<Data,Instance>(test_inst[i], "tst" + to_string(i), test_words[i], false, epoch_pair);
    }
    // Reset the trainer after online learning
    if(epoch == online_epochs_) {
      trainer_ = get_trainer(trainer_id_, learning_rate_, weight_decay_, *mod_);
      trainer_->clipping_enabled = clipping_enabled_;
    }
    model_dropout_prob_ *= model_dropout_decay_;
  }

  cout << "Done training! (s=" << time_.Elapsed() << ")" << endl;

}

// *************** Sanity check code

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


int ModlmTrain::main(int argc, char** argv) {
  po::options_description desc("*** modlm-train (by Graham Neubig) ***");
  desc.add_options()
      ("help", "Produce help message")
      ("clipping_enabled", po::value<bool>()->default_value(true), "Whether to enable clipping or not")
      ("evaluate_frequency", po::value<int>()->default_value(1), "How many times to evaluate for each training epoch")
      ("cnn_mem", po::value<int>()->default_value(512), "Memory used by cnn in megabytes")
      ("cnn_seed", po::value<int>()->default_value(0), "Random seed (default 0 -> changes every time)")
      ("dist_models", po::value<string>()->default_value(""), "Files containing the distribution models")
      ("dropout_models", po::value<string>()->default_value(""), "Which models should be dropped out (zero-indexed ints in comma-delimited groups separated by spaces)")
      ("model_dropout_prob", po::value<float>()->default_value(0.0), "Starting model dropout probability")
      ("model_dropout_decay", po::value<float>()->default_value(1.0), "Model dropout probability decay (1.0 for no decay)")
      ("epochs", po::value<int>()->default_value(300), "Number of epochs")
      ("heuristic", po::value<string>()->default_value(""), "Type of heuristic to use")
      ("layers", po::value<string>()->default_value("ff:50:1"), "Descriptor for hidden layers in format type(ff/rnn/lstm):nodes:layers")
      ("learning_rate", po::value<float>()->default_value(0.1), "Learning rate")
      ("max_minibatch", po::value<int>()->default_value(256), "Max minibatch size")
      ("model_in", po::value<string>()->default_value(""), "If resuming training, read the model in")
      ("model_out", po::value<string>()->default_value(""), "File to write the model to")
      ("online_epochs", po::value<int>()->default_value(-1), "Number of epochs of online learning to perform before switching to batch (-1: only online)")
      ("lstm_dropout_prob", po::value<float>()->default_value(0.0), "How much dropout to cause the LSTM to do")
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
      ("weight_decay", po::value<float>()->default_value(1e-6), "How much weight decay to perform")
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
  evaluate_frequency_ = vm["evaluate_frequency"].as<int>();
  trainer_id_ = vm["trainer"].as<string>();
  learning_rate_ = vm["learning_rate"].as<float>();
  clipping_enabled_ = vm["clipping_enabled"].as<bool>();
  training_type_ = vm["training_type"].as<string>();
  rate_decay_ = vm["rate_decay"].as<float>();
  model_out_file_ = vm["model_out"].as<string>();
  model_dropout_prob_ = vm["model_dropout_prob"].as<float>();
  model_dropout_decay_ = vm["model_dropout_decay"].as<float>();
  lstm_dropout_prob_ = vm["lstm_dropout_prob"].as<float>();
  weight_decay_ = vm["weight_decay"].as<float>();

  // Create a heuristic if using one
  if(vm["heuristic"].as<string>() != "")
    heuristic_ = HeuristicFactory::create_heuristic(vm["heuristic"].as<string>());
  if(vm["whiten"].as<string>() != "")
    whitener_.reset(new Whitener(vm["whiten"].as<string>(), vm["whiten_eps"].as<float>()));

  // Calculate the number of layers
  hidden_spec_ = BuilderSpec(vm["layers"].as<string>());
  if(hidden_spec_.type != "ff" && training_type_ == "agg")
    THROW_ERROR("Only feed-forward networks can be used with aggregated training");

  // Calculate model dropout
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
        sort(dropout_spans_[i].begin(), dropout_spans_[i].end());
      }
      mod_id++;
    }
  }

  cout << "Creating model (s=" << time_.Elapsed() << ")" << endl;

  // Initialize
  mod_.reset(new cnn::Model);
  trainer_ = get_trainer(trainer_id_, learning_rate_, weight_decay_, *mod_);
  trainer_->clipping_enabled = clipping_enabled_;

  float uniform_prob = 1.0/dict_->size();
  log_unk_prob_ = vm["penalize_unk"].as<bool>() ? log(uniform_prob) : 0;

  int num_dist = num_sparse_dist_ + num_dense_dist_;
  if(use_context_) {
    int last_size = num_ctxt_ + word_rep_ * word_hist_;
    // Add the word representation and transformation functions
    if(word_hist_ != 0)
      reps_ = mod_->add_lookup_parameters(dict_->size(), {(unsigned int)word_rep_});
    if(hidden_spec_.layers != 0) {
      builder_ = BuilderFactory::CreateBuilder(hidden_spec_, last_size, *mod_);
      last_size = hidden_spec_.nodes;
    }
    V_ = mod_->add_parameters({(unsigned int)num_dist, (unsigned int)last_size});
  }
  a_ = mod_->add_parameters({(unsigned int)num_dist});

  // Actually perform training
  if(training_type_ == "agg") {
    perform_training<IndexedAggregateDataMap,IndexedAggregateData,IndexedAggregateInstance>();
  } else if(training_type_ == "sent") {
    perform_training<int,IndexedSentenceData,IndexedSentenceInstance>();
  } else {
    THROW_ERROR("Illegal training type");
  }

  return 0;

}

}

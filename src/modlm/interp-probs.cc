
#include <iostream>
#include <fstream>
#include <string>
#include <boost/program_options.hpp>
#include <boost/range/irange.hpp>
#include <boost/algorithm/string.hpp>
#include <dynet/dict.h>
#include <modlm/interp-probs.h>
#include <modlm/macros.h>
#include <modlm/timer.h>
#include <modlm/sentence.h>
#include <modlm/dist-base.h>
#include <modlm/dist-factory.h>
#include <modlm/dict-utils.h>

using namespace std;
using namespace modlm;
namespace po = boost::program_options;

inline void load_probs(const std::string & filename, vector<float> & probs) {
  ifstream in(filename);
  if(!in) THROW_ERROR("Could not open probability file: " << filename);
  float fl;
  while(in >> fl)
    probs.push_back(exp(-fl));
}

inline float accumulate(const vector<vector<float> > & probs, const vector<float> & coeff, vector<float> & coeff_new) {
  size_t num_dists = probs.size();
  vector<float> tmp_prob(num_dists, 0.f);
  size_t num_wds = probs[0].size();
  float loglik = 0.f, tot;
  size_t did;
  assert(coeff.size() == coeff_new.size());
  for(float & c : coeff_new) c = 0.f;
  for(size_t wid = 0; wid < num_wds; ++wid) {
    tot = 0;
    for(did = 0; did < num_dists; ++did)
      tot += (tmp_prob[did] = probs[did][wid] * coeff[did]);
    for(did = 0; did < num_dists; ++did)
      coeff_new[did] += tmp_prob[did]/tot;
    loglik -= log(tot);
  }
  return loglik;
}

int InterpProbs::main(int argc, char** argv) {
  po::options_description desc("*** interp-probs (by Graham Neubig) ***");
  desc.add_options()
    ("help", "Produce help message")
    ("train_files", po::value<string>()->default_value(""), "Pipe-separated probability files used to train interp coefficients")
    ("test_files", po::value<string>()->default_value(""), "Pipe-separated probability files for testing (number must be divisible by number of train_files)")
    ("prob_out", po::value<string>()->default_value(""), "File to write the model to")
    ("epochs", po::value<int>()->default_value(100), "How epochs of training to perform")
    ("verbose", po::value<int>()->default_value(0), "How much verbose output to print")
    ;
  boost::program_options::variables_map vm_;
  po::store(po::parse_command_line(argc, argv, desc), vm_);
  po::notify(vm_);   
  if (vm_.count("help")) {
    cout << desc << endl;
    return 1;
  }

  // Load the probabilities
  boost::split(train_files_, vm_["train_files"].as<string>(), boost::is_any_of("|"));
  if(train_files_.size() < 2) THROW_ERROR("Must have at least two distributions to mix");
  if(vm_["test_files"].as<string>() != "")
    boost::split(test_files_, vm_["test_files"].as<string>(), boost::is_any_of("|"));
  if(test_files_.size() % train_files_.size() != 0)
    THROW_ERROR("Test files_ must be divisible by train files_");
  int num_test = test_files_.size()/train_files_.size();
  vector<vector<float> > train_probs(train_files_.size());
  vector<vector<vector<float> > > test_probs(test_files_.size()/train_files_.size(), train_probs);
  for(size_t fid = 0; fid < train_files_.size(); fid++) {
    load_probs(train_files_[fid], train_probs[fid]);
    if(train_probs[fid].size() != train_probs[0].size())
      THROW_ERROR("Mismatched probability sizes: " << train_files_[0] << " (" << train_probs[0].size() << ") != " << train_files_[fid] << " (" << train_probs[fid].size() << ")");
    for(int tid = 0; tid < num_test; tid++) {
      load_probs(test_files_[tid*train_files_.size()+fid], test_probs[tid][fid]);
      if(test_probs[tid][fid].size() != test_probs[tid][0].size())
        THROW_ERROR("Mismatched probability sizes: " << test_files_[tid*train_files_.size()] << " (" << test_probs[tid][0].size() << ") != " << train_files_[tid*train_files_.size()+fid] << " (" << test_probs[tid][fid].size() << ")");
    }
  }
  const size_t num_dists = train_probs.size();
  vector<float> coeff(num_dists, 1.f/num_dists), coeff_new(num_dists), coeff_tmp(num_dists);

  // Perform EM
  int epochs = vm_["epochs"].as<int>();
  float loglik;
  for(int epoch = 0; epoch < epochs; ++epoch) {
    // Calculate and print training
    loglik = accumulate(train_probs, coeff, coeff_new);
    cerr << "trn  epoch " << epoch+1 << ": ppl=" << exp(loglik/train_probs[0].size()) << ", coeff:"; for(float f : coeff) cerr << ' ' << f; cerr << endl;
    // Calculate and print all tests
    for(size_t tid = 0; tid < test_probs.size(); ++tid) {
      loglik = accumulate(test_probs[tid], coeff, coeff_tmp);
      cerr << "tst" << tid << " epoch " << epoch+1 << ": ppl=" << exp(loglik/test_probs[tid][0].size()) << endl;
    }
    // Create new coefficients
    float tot = 0.f;
    size_t did;
    for(did = 0; did < num_dists; ++did)
      tot += coeff_new[did];
    for(did = 0; did < num_dists; ++did)
      coeff[did] = coeff_new[did]/tot;
  }

  // Open output file
  string prob_out_file = vm_["prob_out"].as<string>();
  if(prob_out_file != "") {
    ofstream prob_out(prob_out_file);
    if(!prob_out)
      THROW_ERROR("Could not write to output file: " << prob_out_file);
    prob_out << coeff[0];
    for(size_t i = 1; i < coeff.size(); i++)
      prob_out << ' ' << coeff[i];
    prob_out << endl;
  }


  return 0;
}

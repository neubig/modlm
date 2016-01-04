#include <cmath>
#include <modlm/heuristic.h>
#include <modlm/macros.h>

using namespace std;
using namespace modlm;

std::vector<float> HeuristicAbs::smooth(int num_dists, const std::vector<float> & ctxts) {
  if(ctxts.size() % 4 != 0 || num_dists*4 < ctxts.size())
    THROW_ERROR("Absolute discounting heuristic expects 4 context features for each dist");
  vector<float> ret(num_dists, 0.0);
  float left = 1.0;
  int ngram_dists = ctxts.size()/4;
  for(int i = ngram_dists - 1; i >= 0; i--) {
    if(ctxts[i*4] == 1.0) continue;
    // Discounted divided by total == amount for this dist
    float my_prob = exp(ctxts[i*4+3]-ctxts[i*4+1]);
    ret[i] = my_prob * left;
    left *= (1-my_prob);
  }
  for(int i = ngram_dists; i < num_dists; i++)
    ret[i] = left/(num_dists-ngram_dists);
  return ret;
}

std::vector<float> HeuristicWb::smooth(int num_dists, const std::vector<float> & ctxts) {
  if(ctxts.size() % 3 != 0 || num_dists*3 < ctxts.size())
    THROW_ERROR("Witten bell heuristic expects 3 context features for each dist");
  vector<float> ret(num_dists, 0.0);
  float left = 1.0;
  int ngram_dists = ctxts.size()/3;
  for(int i = ngram_dists-1; i >= 0; i--) {
    if(ctxts[i*3] == 1.0) continue;
    // Discounted divided by total == amount for this dist
    float word = exp(ctxts[i*3+1]), uniq = exp(ctxts[i*3+2]);
    float my_prob = word/(word+uniq);
    ret[i] = my_prob * left;
    left *= (1-my_prob);
  }
  for(int i = ngram_dists; i < num_dists; i++)
    ret[i] = left/(num_dists-ngram_dists);
  return ret;
}


HeuristicPtr HeuristicFactory::create_heuristic(const std::string & sig) {
  if(sig == "abs") {
    return HeuristicPtr(new HeuristicAbs());
  } else if(sig == "wb") {
    return HeuristicPtr(new HeuristicWb());
  } else {
    THROW_ERROR("Bad heuristic signature: " << sig);
  }
}

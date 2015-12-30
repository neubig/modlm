#pragma once

#include <vector>
#include <memory>

namespace modlm {

class Heuristic {
public:
  Heuristic() { }
  virtual ~Heuristic() { }

  virtual std::vector<float> smooth(int num_dists, const std::vector<float> & ctxts) = 0;
};

typedef std::shared_ptr<Heuristic> HeuristicPtr;

class HeuristicAbs : public Heuristic {
public:

  virtual std::vector<float> smooth(int num_dists, const std::vector<float> & ctxts) override;
};

class HeuristicWb : public Heuristic {
public:

  virtual std::vector<float> smooth(int num_dists, const std::vector<float> & ctxts) override;
};

class HeuristicFactory {

public:
  static HeuristicPtr create_heuristic(const std::string & sig);

};

}

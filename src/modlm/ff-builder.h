#pragma once

#include <dynet/rnn.h>
#include <dynet/dynet.h>
#include <dynet/rnn-state-machine.h>
#include <dynet/expr.h>

using namespace dynet::expr;

namespace dynet {

class Model;

struct FFBuilder : public RNNBuilder {
  FFBuilder() = default;
  explicit FFBuilder(unsigned layers,
                     unsigned input_dim,
                     unsigned hidden_dim,
                     Model* model);

 protected:
  virtual Expression set_h_impl(int prev, const std::vector<Expression>& h_new) override;
  void new_graph_impl(ComputationGraph& cg) override;
  void start_new_sequence_impl(const std::vector<Expression>& h_0) override;
  Expression add_input_impl(int prev, const Expression& x) override;

 public:
  Expression add_auxiliary_input(const Expression& x, const Expression &aux);

  void set_dropout(float d) { dropout_rate = d; }

  Expression back() const override { return h0.back(); }
  std::vector<Expression> final_h() const override { return h0; }
  std::vector<Expression> final_s() const override { return final_h(); }

  std::vector<Expression> get_h(RNNPointer i) const override { return h0; }
  std::vector<Expression> get_s(RNNPointer i) const override { return get_h(i); }
  void copy(const RNNBuilder & params) override;

  unsigned num_h0_components() const override { return 0; }

 private:
  // first index is layer, then x2h hb
  std::vector<std::vector<Parameter> > params;

  // first index is layer, then x2h hb
  std::vector<std::vector<Expression>> param_vars;

  // initial value of h
  // defaults to zero matrix input
  std::vector<Expression> h0;

  float dropout_rate;

  unsigned layers;
};

} // namespace dynet


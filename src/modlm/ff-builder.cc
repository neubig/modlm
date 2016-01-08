#include <modlm/ff-builder.h>
#include <cnn/rnn.h>
#include <cnn/nodes.h>
#include <cnn/expr.h>
#include <string>
#include <cassert>
#include <vector>
#include <iostream>

using namespace std;
using namespace cnn::expr;

namespace cnn {

enum { X2H=0, HB };

FFBuilder::FFBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Model* model) : layers(layers) {
  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    Parameters* p_x2h = model->add_parameters({hidden_dim, layer_input_dim});
    Parameters* p_hb = model->add_parameters({hidden_dim});
    params.push_back({p_x2h, p_hb});
    layer_input_dim = hidden_dim;
  }
}

void FFBuilder::new_graph_impl(ComputationGraph& cg) {
  param_vars.clear();
  for (unsigned i = 0; i < layers; ++i) {
    Parameters* p_x2h = params[i][X2H];
    Parameters* p_hb = params[i][HB];
    Expression i_x2h =  parameter(cg,p_x2h);
    Expression i_hb =  parameter(cg,p_hb);
    vector<Expression> vars = {i_x2h, i_hb};

    param_vars.push_back(vars);
  }
}

void FFBuilder::start_new_sequence_impl(const vector<Expression>& h_0) {
  assert(h_0.size() == 0);
}

Expression FFBuilder::add_input_impl(int prev, const Expression &in) {

  Expression x = in;

  if(dropout_rate) x = dropout(x, dropout_rate);

  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];

    // y <--- f(x)
    Expression y = affine_transform({vars[1], vars[0], x});

    // x <--- tanh(y)
    x = tanh(y);

    if(dropout_rate) x = dropout(x, dropout_rate);

  }
  return x;
}

Expression FFBuilder::add_auxiliary_input(const Expression &in, const Expression &aux) {
  assert(false);
}

void FFBuilder::copy(const RNNBuilder & rnn) {
  const FFBuilder & rnn_simple = (const FFBuilder&)rnn;
  assert(params.size() == rnn_simple.params.size());
  for(size_t i = 0; i < rnn_simple.params.size(); ++i) {
      params[i][0]->copy(*rnn_simple.params[i][0]);
      params[i][1]->copy(*rnn_simple.params[i][1]);
  }
}

} // namespace cnn

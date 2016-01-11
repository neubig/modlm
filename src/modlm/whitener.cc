#include <iostream>
#include <boost/range/irange.hpp>
#include <Eigen/Core>
#include <Eigen/Eigenvalues> 
#include <modlm/whitener.h>
#include <modlm/macros.h>

using namespace std;
using namespace modlm;
using namespace Eigen;

// Find the transformation matrix for whitening in Eigen column-major format
void Whitener::calc_matrix(const std::vector<std::vector<float> > & data, const std::vector<int> & use_in_training) {

  if(type_ == "") return;
  bool do_mean = (type_ == "mean");

  int data_rows = 0, my_data_row = 0, data_cols = data[0].size();

  // Convert the matrix into Eigen
  for(int use : use_in_training)
    if(use > 0)
      data_rows += (do_mean ? 1 : use);
  MatrixXf X(data_rows, data_cols);
  int DEBUG_HERE = 0;
  for(int i : boost::irange(0, (int)data.size())) {
    const auto & out_val = data[i];
    int multiplier = use_in_training[i];
    if(!do_mean) {
      for(int k = 0; k < multiplier; k++) {
        for(int j : boost::irange(0, data_cols))
          X(my_data_row,j) = out_val[j];
        my_data_row++;
      }
    } else if(multiplier != 0) {
      for(int j : boost::irange(0, data_cols)) 
        X(my_data_row,j) = out_val[j] * multiplier;
      my_data_row++;
    }
  }
  // cerr << "---------- X" << endl << X << endl;
  // Calculate the mean and center
  mean_vec_.resize(data_cols);
  Map<RowVectorXf> X_mean(&mean_vec_[0], data_cols);
  X_mean = X.colwise().mean();

  if(do_mean) return;

  // Calculate the covariance and perform whitening
  X.rowwise() -= X_mean;
  MatrixXf Xcov = (X.transpose() * X)/(float)data_rows;

  // ZCA
  if(type_ == "zca") {
    JacobiSVD<MatrixXf> svd(Xcov, ComputeThinU);
    // Convert into a diagonal matrix
    ArrayXf ev = svd.singularValues();
    DiagonalMatrix<float,Dynamic> D(data_cols);
    D.diagonal() = 1/(ev+epsilon_).sqrt();
    // Create the whitening matrix
    rotation_vec_.resize(data_cols * data_cols);
    Map<MatrixXf, Aligned> W(&rotation_vec_[0], data_cols, data_cols);
    W = ((svd.matrixU() * D) * svd.matrixU().transpose()); 
  // PCA
  } else if(type_ == "pca") {
    SelfAdjointEigenSolver<MatrixXf> es(Xcov);
    // Convert into a diagonal matrix
    ArrayXf ev = es.eigenvalues();
    DiagonalMatrix<float,Dynamic> D(data_cols);
    D.diagonal() = 1/(ev+epsilon_).sqrt();
    // Create the whitening matrix
    rotation_vec_.resize(data_cols * data_cols);
    Map<MatrixXf, Aligned> W(&rotation_vec_[0], data_cols, data_cols);
    W = ((es.eigenvectors() * D) * es.eigenvectors().transpose()); 
  } else {
    THROW_ERROR("Illegal whitener type: " << type_);
  }
}
// Perform whitening
void Whitener::whiten(std::vector<std::vector<float> > & data) {
  if(type_ == "") return;
  // Create the data vector
  int data_rows = data.size();
  int data_cols = mean_vec_.size();
  Map<const RowVectorXf> X_mean(&mean_vec_[0], data_cols);
  // Perform whitening on each example
  if(type_ == "mean") {
    for(auto & my_data : data) {
      Map<RowVectorXf> x(&my_data[0], data_cols);
      x = (x - X_mean);
    }
  } else {
    Map<const MatrixXf> W(&rotation_vec_[0], data_cols, data_cols);
    for(auto & my_data : data) {
      Map<RowVectorXf> x(&my_data[0], data_cols);
      x = (x - X_mean) * W;
    }
  }
}

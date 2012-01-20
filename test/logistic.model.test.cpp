#include <iostream>
#include <vector>
#include <armadillo>
#include <cppbugs/cppbugs.hpp>
#include <algorithm>

using namespace arma;
using namespace cppbugs;
using std::cout;
using std::endl;

int main() {
  const int NR = 1e2;
  const int NC = 2;
  const ivec size = 1000 * ones<ivec>(NR);
  const mat real_b = mat("0.1; 1.0");
  mat X = mat(NR,NC);
  X.col(0).fill(1);
  X.col(1) = randn<mat>(NR,1);
  const ivec y = conv_to<ivec>::from(size / (1+exp(-X*real_b)));

  vec b(randn<vec>(NC));
  vec y_hat, p_hat;
  double rsq;

  std::function<void ()> model = [&]() {
    p_hat = 1/(1+exp(-X*b));
    y_hat = p_hat % size;
    rsq = as_scalar(1 - var(y - y_hat) / var(y));
  };

  MCModel m(model);
  m.normal(b).dnorm(0.0, 0.0001);
  m.binomial(y).dbinom(size,p_hat);
  m.deterministic(rsq);

  int iterations = 1e5;
  m.sample(iterations, 1e4, 1e4, 10);

  cout << "b: " << endl << m.getNode(b).mean();
  cout << "R^2: " << m.getNode(rsq).mean() << endl;
  cout << "samples: " << m.getNode(b).history.size() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;
  return 0;
};

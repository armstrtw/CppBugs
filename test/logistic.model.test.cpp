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
  vec p_hat = 1/(1+exp(-X*b));
  vec y_hat = p_hat % size;
  double rsq;

  std::function<void ()> model = [&]() {
    p_hat = 1/(1+exp(-X*b));
    y_hat = p_hat % size;
    rsq = as_scalar(1 - var(y - y_hat) / var(y));
  };

  MCModel<boost::minstd_rand> m(model);
  m.link<Normal>(b, 0, 0.001);
  m.link<Deterministic>(p_hat);
  m.link<Deterministic>(y_hat);
  m.link<ObservedBinomial>(y, size, p_hat);
  m.link<Deterministic>(rsq);

  // things to track
  std::vector<vec>& b_hist = m.track<std::vector>(b);
  std::vector<double>& rsq_hist = m.track<std::vector>(rsq);

  int iterations = 1e5;
  m.sample(iterations, 1e4, 1e4, 10);

  cout << "b (actual):" << endl << real_b;
  cout << "b: " << endl << mean(b_hist.begin(),b_hist.end());
  cout << "R^2: " << mean(rsq_hist.begin(),rsq_hist.end()) << endl;
  cout << "samples: " << b_hist.size() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;
  return 0;
}

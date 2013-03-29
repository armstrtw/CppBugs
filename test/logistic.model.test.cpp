#include <iostream>
#include <vector>
#include <armadillo>
#include <cppbugs/cppbugs.hpp>
#include <cppbugs/deterministics/mcmc.logistic.hpp>

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

  MCModel<boost::minstd_rand> m;
  m.link<Normal>(b, 0, 0.001);
  m.link<Logistic>(p_hat, X, b);
  m.link<ObservedBinomial>(y, size, p_hat);


  // things to track
  std::vector<vec>& b_hist = m.track<std::vector>(b);

  m.tune(1e4,100);
  m.tune_global(1e4,100);
  m.burn(1e4);
  m.sample(1e5, 10);

  cout << "b (actual):" << endl << real_b;
  cout << "b: " << endl << mean(b_hist.begin(),b_hist.end());
  cout << "samples: " << b_hist.size() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;
  return 0;
}

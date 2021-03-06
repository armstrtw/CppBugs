#include <iostream>
#include <vector>
#include <armadillo>
#include <cppbugs/cppbugs.hpp>
#include <cppbugs/mcmc.boost.rng.hpp>
#include <cppbugs/deterministics/mcmc.linear.hpp>
#include <cppbugs/deterministics/mcmc.rsquared.hpp>

using namespace arma;
using namespace cppbugs;
using std::cout;
using std::endl;

int main() {
  const int NR = 1e2;
  const int NC = 2;
  const mat y = randn<mat>(NR,1) + 10;
  mat X = mat(NR,NC);
  X.col(0).fill(1);
  X.col(1) = y + randn<mat>(NR,1)/2 - 10;

  vec coefs;
  solve(coefs, X, y);
  vec err = y - X*coefs;

  vec b = randn<vec>(2);
  mat y_hat = X * b;
  double rsq(0);
  double tau_y(1);

  /*
  std::function<void ()> model = [&]() {
    y_hat = X * b;
    rsq = as_scalar(1 - var(y - y_hat) / var(y));
  };
  */

  BoostRng<boost::minstd_rand> rng;
  MCModel m(rng);

  m.link<Normal>(b, 0, 0.001);
  m.link<Uniform>(tau_y, 0, 100);
  m.link<Linear>(y_hat, X, b);
  m.link<ObservedNormal>(y, y_hat, tau_y);
  m.link<Rsquared>(rsq,y,y_hat);

  std::vector<vec>& b_hist = m.track<std::vector>(b);
  std::vector<double>& tau_y_hist = m.track<std::vector>(tau_y);
  std::vector<double>& rsq_hist = m.track<std::vector>(rsq);

  m.tune(1e4,100);
  m.tune_global(1e4,100);
  m.burn(1e4);
  m.sample(1e5, 10);

  cout << "lm coefs" << endl << coefs;
  cout << "err sd: " << stddev(err,0) << endl;;
  cout << "err tau: " << pow(stddev(err,0),-2) << endl;

  cout << "b: " << endl << mean(b_hist.begin(),b_hist.end()) << endl;
  cout << "tau_y: " << mean(tau_y_hist.begin(),tau_y_hist.end()) << endl;
  cout << "R^2: " << mean(rsq_hist.begin(),rsq_hist.end()) << endl;
  cout << "samples: " << b_hist.size() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;

  return 0;
};

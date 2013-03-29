#include <iostream>
#include <vector>
#include <armadillo>
#include <boost/random.hpp>
#include <cppbugs/cppbugs.hpp>
#include <cppbugs/mcmc.model.hpp>
#include <cppbugs/deterministics/mcmc.linear.grouped.hpp>
#include <cppbugs/deterministics/mcmc.rsquared.hpp>


using namespace arma;
using namespace cppbugs;
using std::cout;
using std::endl;

int main() {
  const unsigned int NR = 100;
  const unsigned int NC = 2;
  const unsigned int J = 3;

  mat X = randn<mat>(NR,NC);
  mat y = randn<mat>(NR,1);
  const mat& y_const(y);

  X.col(0).fill(1.0);

  // create fake groups
  uvec groups(NR);
  for(unsigned int i = 0; i < NR; i++) {
    groups[i] = i % J;
  }

  // shift y's by group sums
  vec group_shift(J);
  for(unsigned int i = 0; i < J; i++) {
    group_shift[i] = (i + 1) * 10;
  }
  cout << "group_shift" << endl << group_shift;

  // do the shift on the data
  for(unsigned int i = 0; i < NR; i++) {
    y[i] += group_shift[ groups[i] ];
  }

  vec coefs;
  solve(coefs, X, y);
  vec err = y - X*coefs;

  mat b(randn<mat>(J,NC));
  double tau_y(1), rsq(0);
  mat y_hat;

  MCModel<boost::minstd_rand> m;
  m.link<Normal>(b, 0, 0.001);
  m.link<Uniform>(tau_y, 0, 100.);
  m.link<LinearGrouped>(y_hat, X, b, groups);
  m.link<ObservedNormal>(y_const, y_hat, tau_y);
  m.link<Rsquared>(rsq,y,y_hat);

  // things to track
  std::vector<mat>& b_hist = m.track<std::vector>(b);
  std::vector<double>& tau_y_hist = m.track<std::vector>(tau_y);
  std::vector<double>& rsq_hist = m.track<std::vector>(rsq);

  m.tune(1e4,100);
  m.tune_global(1e4,100);
  m.burn(1e4);
  m.sample(1e5, 5);

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

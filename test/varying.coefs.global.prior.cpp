#include <iostream>
#include <vector>
#include <armadillo>
#include <boost/random.hpp>
#include <cppbugs/cppbugs.hpp>
#include <cppbugs/mcmc.boost.rng.hpp>
#include <cppbugs/deterministics/mcmc.linear.grouped.hpp>
#include <cppbugs/deterministics/mcmc.rsquared.hpp>
#include <cppbugs/mcmc.model.hpp>

using namespace arma;
using namespace cppbugs;
using std::cout;
using std::endl;

typedef arma::subview_elem2<double, arma::Mat<uword>, arma::Mat<uword> > replicatedT;

int main() {
  const uword NR = 100;
  const uword NC = 2;
  const uword J = 3;

  mat X = randn<mat>(NR,NC);
  mat y = randn<mat>(NR,1);
  const mat& y_const(y);

  X.col(0).fill(1.0);

  // create fake groups
  uvec groups(NR);
  for(uword i = 0; i < NR; i++) {
    groups[i] = i % J;
  }

  // shift y's by group sums
  vec group_shift(J);
  for(unsigned int i = 0; i < J; i++) {
    group_shift[i] = (i + 1) * 10;
  }
  cout << "group_shift" << endl << group_shift;

  // do the shift on the data
  for(uword i = 0; i < NR; i++) {
    y[i] += group_shift[ groups[i] ];
  }

  vec coefs;
  solve(coefs, X, y);
  vec err = y - X*coefs;

  //vec rowdup(ones<vec>(J));
  uvec rowdup(zeros<uvec>(J));

  mat b_mu(randn<mat>(1,NC));
  mat b_tau(randu<mat>(1,NC));
  mat b(randn<mat>(J,NC));
  double tau_y(1);
  mat y_hat;
  double rsq;
  replicatedT b_mu_full_rnk = b_mu.rows(rowdup);
  replicatedT b_tau_full_rnk = b_tau.rows(rowdup);

  BoostRng<boost::minstd_rand> rng;
  MCModel m(rng);

  m.link<Normal>(b_mu, 0, 0.001);
  m.link<Uniform>(b_tau, 0, 100);
  m.link<Normal>(b, b_mu_full_rnk, b_tau_full_rnk);
  m.link<Uniform>(tau_y, 0, 100);
  m.link<LinearGrouped>(y_hat, X, b, groups);
  m.link<ObservedNormal>(y_const, y_hat, tau_y);
  m.link<Rsquared>(rsq,y,y_hat);

  // things to track
  std::vector<mat>& b_hist = m.track<std::vector>(b);
  std::vector<mat>& b_mu_hist = m.track<std::vector>(b_mu);
  std::vector<mat>& b_tau_hist = m.track<std::vector>(b_tau);
  std::vector<double>& tau_y_hist = m.track<std::vector>(tau_y);
  std::vector<double>& rsq_hist = m.track<std::vector>(rsq);

  m.tune(1e4,100);
  m.tune_global(1e4,100);
  m.burn(1e4);
  m.sample(1e5, 5);

  cout << "lm coefs" << endl << coefs;
  cout << "err sd: " << stddev(err,0) << endl;;
  cout << "err tau: " << pow(stddev(err,0),-2) << endl;

  cout << "b: " << endl << mean(b_hist.begin(),b_hist.end());
  cout << "b_mu: " << endl << mean(b_mu_hist.begin(),b_mu_hist.end());
  cout << "b_tau: " << endl << mean(b_tau_hist.begin(),b_tau_hist.end());
  cout << "tau_y: " << mean(tau_y_hist.begin(),tau_y_hist.end()) << endl;
  cout << "R^2: " << mean(rsq_hist.begin(),rsq_hist.end()) << endl;
  cout << "samples: " << b_hist.size() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;
  return 0;
};

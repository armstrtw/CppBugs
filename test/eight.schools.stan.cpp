#include <iostream>
#include <vector>
#include <functional>
#include <armadillo>
#include <boost/random.hpp>
#include <cppbugs/cppbugs.hpp>
#include <cppbugs/mcmc.model.hpp>
#include <cppbugs/deterministics/mcmc.linear.with.const.hpp>


using namespace arma;
using namespace cppbugs;
using std::cout;
using std::endl;

int main() {

  const int J = 8;
  const vec sigma_y({15,10,16,11,9,11,10,18});
  const vec tau_y = pow(sigma_y,-2);
  const vec y({28,  8, -3,  7, -1,  1, 18, 12});

  double mu(0);
  double tau(100);
  vec eta = randn<vec>(J);
  vec theta = mu + tau * eta;

  MCModel<boost::minstd_rand> m;

  // noninformative prior on mu
  m.link<Normal>(mu, 0.0, 1.0E-6);

  // noninformative prior on tau
  m.link<Uniform>(tau, 0, 1000);
  m.link<Normal>(eta, 0, 1);
  m.link<LinearWithConst>(theta,eta,mu,tau);
  m.link<ObservedNormal>(y, theta, tau_y); //*

  // things to track
  std::vector<vec>& eta_hist = m.track<std::vector>(eta);
  std::vector<vec>& theta_hist = m.track<std::vector>(theta);

  m.tune(1e3,100);
  m.tune_global(1e3,100);
  m.burn(1e5);
  m.sample(1e5, 1);

  cout << "eta:" << endl << mean(eta_hist.begin(),eta_hist.end()) << endl;
  cout << "theta:" << endl << mean(theta_hist.begin(),theta_hist.end()) << endl;
  cout << "samples: " << theta_hist.size() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;
  return 0;
}

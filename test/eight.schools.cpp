#include <iostream>
#include <vector>
#include <functional>
#include <armadillo>
#include <boost/random.hpp>
#include <cppbugs/cppbugs.hpp>
#include <cppbugs/mcmc.model.hpp>


using namespace arma;
using namespace cppbugs;
using std::cout;
using std::endl;

int main() {

  const int J = 8;
  const vec sigma_y({15,10,16,11,9,11,10,18});
  const vec tau_y = pow(sigma_y,-2);
  const vec y({28,  8, -3,  7, -1,  1, 18, 12});

  double mu_theta(0);
  double sigma_theta(1);
  double tau_theta = pow(sigma_theta,-2);
  vec theta = randn<vec>(J);

  std::function<void ()> model = [&]() {
    tau_theta = pow(sigma_theta,-2);
  };

  MCModel<boost::minstd_rand> m(model);

  // noninformative prior on mu
  m.link<Normal>(mu_theta, 0.0, 1.0E-6);

  // noninformative prior on sigma
  m.link<Uniform>(sigma_theta, 0, 1000);

  m.link<Normal>(theta,mu_theta,tau_theta);
  m.link<ObservedNormal>(y, theta, tau_y);

  // things to track
  std::vector<vec>& theta_hist = m.track<std::vector>(theta);

  m.tune(1e4,100);
  m.tune_global(1e4,100);
  m.burn(1e4);
  m.sample(1e5, 5);

  cout << "theta:" << endl << mean(theta_hist.begin(),theta_hist.end()) << endl;
  cout << "samples: " << theta_hist.size() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;
  return 0;
}

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

  // setup data
  const vec age({13, 14, 14,12, 9, 15, 10, 14, 9, 14, 13, 12, 9, 10, 15, 11, 15, 11, 7,
        13, 13, 10, 9, 6, 11, 15, 13, 10, 9, 9, 15, 14, 14, 10, 14, 11, 13, 14, 10});
  const vec price_r({2950, 2300, 3900, 2800, 5000, 2999, 3950, 2995, 4500, 2800, 1990, 3500, 5100, 3900, 2900,
        4950, 2000, 3400, 8999, 4000, 2950, 3250, 3950, 4600, 4500, 1600, 3900, 4200, 6500, 3500, 2999, 2600, 3250, 2500, 2400, 3990, 4600, 450,4700});
  const vec price(price_r/1000);
  const int NR = price.n_rows;

  // fit linear model
  const mat X = join_rows(ones<vec>(NR),age);
  vec coefs;
  solve(coefs, X, price);
  vec err = price - X*coefs;  
  double a(coefs[0]), b(coefs[1]), tau(1);
  mat y_hat;

  MCModel<boost::minstd_rand> m;
  m.link<Normal>(a, 0, 0.001);
  m.link<Normal>(b, 0, 0.001);
  m.link<Gamma>(tau, 0.001, 0.001);
  m.link<LinearWithConst>(y_hat,age,a,b);
  m.link<ObservedNormal>(price, y_hat, tau);

  // things to track
  std::vector<double>& a_hist = m.track<std::vector>(a);
  std::vector<double>& b_hist = m.track<std::vector>(b);
  std::vector<double>& tau_hist = m.track<std::vector>(tau);

  m.tune(1e4,100);
  m.tune_global(1e4,100);
  m.burn(1e4);
  m.sample(1e5, 5);

  cout << "lm coefs" << endl << coefs;
  cout << "err sd: " << stddev(err,0) << endl;;
  cout << "err tau: " << pow(stddev(err,0),-2) << endl;

  cout << "a: " << mean(a_hist.begin(),a_hist.end()) << endl;
  cout << "b: " << mean(b_hist.begin(),b_hist.end()) << endl;
  cout << "tau: " << mean(tau_hist.begin(),tau_hist.end()) << endl;
  cout << "samples: " << a_hist.size() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;
  return 0;
}

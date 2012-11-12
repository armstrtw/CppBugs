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
  const double zero(0), one_e1(0.1), one_e3(0.001);

  // setup data
  double ageraw[] = {13, 14, 14,12, 9, 15, 10, 14, 9, 14, 13, 12, 9, 10, 15, 11, 15, 11, 7,
                     13, 13, 10, 9, 6, 11, 15, 13, 10, 9, 9, 15, 14, 14, 10, 14, 11, 13, 14, 10};
  double priceraw[] = {2950, 2300, 3900, 2800, 5000, 2999, 3950, 2995, 4500, 2800, 1990, 3500, 5100, 3900, 2900,
                       4950, 2000, 3400, 8999, 4000, 2950, 3250, 3950, 4600, 4500, 1600, 3900, 4200, 6500, 3500, 2999, 2600, 3250, 2500, 2400, 3990, 4600, 450,4700};
  const int NR = 39;
  const mat age(ageraw,39,1);
  const mat price_r(priceraw,39,1);
  const mat price(price_r/1000);

  // fit linear model
  mat icept(NR,1); icept.fill(1);
  const mat X = join_rows(icept,age);
  vec coefs;
  solve(coefs, X, price);
  vec err = price - X*coefs;


  double a, b, tau;
  mat y_hat = a + b * age;
  std::function<void ()> model = [&]() {
    y_hat = a + b * age;
  };

  MCModel<boost::minstd_rand> m(model);
  m.link<Normal>(a, zero, one_e3);
  m.link<Normal>(b, zero, one_e3);
  m.link<Gamma>(tau, one_e1, one_e1);
  m.link<ObservedNormal>(price, y_hat, tau);

  // things to track
  std::vector<double>& a_hist = m.track<std::vector>(a);
  std::vector<double>& b_hist = m.track<std::vector>(b);
  std::vector<double>& tau_hist = m.track<std::vector>(tau);

  int iterations = 1e5;
  m.sample(iterations, 1e4, 1e4, 5);
  cout << "lm coefs" << endl << coefs;
  cout << "err sd: " << stddev(err,0) << endl;;
  cout << "err tau: " << pow(stddev(err,0),-2) << endl;

  cout << "a: " << std::accumulate(a_hist.begin(),a_hist.end(),0.) / a_hist.size() << endl;
  cout << "b: " << std::accumulate(b_hist.begin(),b_hist.end(),0.) / b_hist.size() << endl;
  cout << "tau: " << std::accumulate(tau_hist.begin(),tau_hist.end(),0.) / tau_hist.size() << endl;
  cout << "samples: " << a_hist.size() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;
  return 0;
};

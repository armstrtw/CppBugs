#include <iostream>
#include <vector>
#include <armadillo>
#include <cppbugs/cppbugs.hpp>

using namespace arma;
using namespace cppbugs;
using std::cout;
using std::endl;

int main() {
  const double zero(0),one_hundred(100),one_e3(0.001);

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
  mat y_hat;
  double rsq(0);
  double tau_y(1);

  std::function<void ()> model = [&]() {
    y_hat = X * b;
    rsq = as_scalar(1 - var(y - y_hat) / var(y));
  };

  MCModel<boost::minstd_rand> m(model);

  m.track<Normal>(b).dnorm(zero, one_e3);
  m.track<Uniform>(tau_y).dunif(zero,one_hundred);
  m.track<ObservedNormal>(y).dnorm(y_hat,tau_y);
  m.track<Deterministic>(rsq);

  int iterations = 1e5;
  m.sample(iterations, 1e4, 1e4, 10);
  cout << "lm coefs" << endl << coefs;
  cout << "err sd: " << stddev(err,0) << endl;;
  cout << "err tau: " << pow(stddev(err,0),-2) << endl;
  cout << "b: " << endl << m.getNode(b).mean();
  cout << "tau_y: " << m.getNode(tau_y).mean() << endl;
  cout << "R^2: " << m.getNode(rsq).mean() << endl;
  cout << "samples: " << m.getNode(b).history.size() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;

  return 0;
};

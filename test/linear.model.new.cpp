#include <iostream>
#include <vector>
#include <armadillo>
#include <boost/random.hpp>
#include <cppbugs/cppbugs.hpp>
#include <cppbugs/mcmc.model.hpp>

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

  Normal<vec> b(randn<vec>(2));
  b.dnorm(0.0, 0.0001);

  Deterministic<mat> y_hat([&]() { return X * b.value; });
  //cout << y_hat.value;

  Deterministic<double >rsq([&]() { return  as_scalar(1 - var(y - y_hat.value) / var(y)); });

  Uniform<double> tau_y(1);
  tau_y.dunif(0.,100.);

  Normal<mat> likelihood(y,true);
  likelihood.dnorm(y_hat,tau_y);
  // cout << "likelihood mu:" << &likelihood.mu_ << endl;
  // cout << "likelihood tau:" << &likelihood.mu_ << endl;

  //std::vector<MCMCObject*> nodes = {&b, &y_hat, &tau_y, &likelihood};
  MCModel m({&b, &y_hat, &tau_y, &likelihood, &rsq});
  int iterations = 1e5;
  m.sample(iterations, 1e4, 1e4, 10);
  cout << "lm coefs" << endl << coefs;
  cout << "err sd: " << stddev(err,0) << endl;;
  cout << "err tau: " << pow(stddev(err,0),-2) << endl;
  cout << "b: " << endl << b.mean();
  cout << "tau_y: " << tau_y.mean() << endl;
  cout << "R^2: " << rsq.mean() << endl;
  cout << "samples: " << b.history.size() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;

  return 0;
};

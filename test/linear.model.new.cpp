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
  mat y = randn<mat>(NR,1) + 10;
  mat X = mat(NR,NC);
  X.col(0).fill(1);
  X.col(1) = y + randn<mat>(NR,1)/2 - 10;

  vec coefs;
  solve(coefs, X, y);
  vec err = y - X*coefs;

  vec b = randn<vec>(2);
  Normal<vec> b_lik(b);
  //Deterministic<mat> y_hat;
  mat y_hat;
  double rsq(0);
  Deterministic<double> rsq_d(rsq);

  double tau_y;
  Uniform<double> tau_y_lik(tau_y);
  Normal<mat> likelihood(y,true);

  b_lik.dnorm(0.0, 0.0001);
  tau_y_lik.dunif(0,100);
  likelihood.dnorm(y_hat,tau_y);

  std::function<void ()> model = [&]() {
    y_hat = X * b;
    rsq = as_scalar(1 - var(y - y_hat) / var(y));
  };

  //MCModel m({&b_lik, &tau_y_lik, &likelihood, &rsq},model);
  MCModel m({&b_lik, &tau_y_lik, &likelihood, &rsq_d},model);
  int iterations = 1e5;
  m.sample(iterations, 1e4, 1e4, 10);
  cout << "lm coefs" << endl << coefs;
  cout << "err sd: " << stddev(err,0) << endl;;
  cout << "err tau: " << pow(stddev(err,0),-2) << endl;
  cout << "b: " << endl << b;
  cout << "b: " << endl << b_lik.mean();
  cout << "tau_y: " << tau_y_lik.mean() << endl;
  cout << "R^2: " << rsq_d.mean() << endl;
  cout << "samples: " << b_lik.history.size() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;

  return 0;
};

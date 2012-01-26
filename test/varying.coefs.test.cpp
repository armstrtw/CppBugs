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
  const unsigned int NR = 100;
  const unsigned int NC = 2;
  const unsigned int J = 3;

  mat X = randn<mat>(NR,NC);
  mat y = randn<mat>(NR,1);
  const mat& y_const(y);

  X.col(0).fill(1.0);

  // create fake groups
  ivec groups(NR);
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

  mat permutation_matrix(NR,J);
  permutation_matrix.fill(0.0);
  for(unsigned int i = 0; i < groups.n_elem; i++) {
    permutation_matrix(i,groups[i]) = 1.0;
  }

  mat b(randn<mat>(J,NC));
  double tau_y, rsq;
  mat y_hat;

  std::function<void ()> model = [&] {
    y_hat = sum(X % (permutation_matrix * b),1);
    rsq = as_scalar(1 - var(y - y_hat) / var(y));
  };

  MCModel<boost::minstd_rand> m(model);
  m.track<Normal>(b).dnorm(0.0, 0.0001);
  m.track<Uniform>(tau_y).dunif(0,100);
  m.track<ObservedNormal>(y_const).dnorm(y_hat,tau_y);
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

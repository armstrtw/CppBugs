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

class TestModel: public MCModel {
public:
  const mat& y; // given
  const mat& X; // given
  const ivec& groups; // given
  mat permutation_matrix;

  Normal<mat> b;
  Uniform<double> tau_y;
  Deterministic<mat> y_hat;
  Normal<mat> likelihood;
  Deterministic<double> rsq;

  TestModel(const mat& y_,const mat& X_,const ivec& groups_, int NG): y(y_), X(X_), groups(groups_),
                                                                      permutation_matrix(X_.n_rows,NG),
                                                                      b(randn<mat>(NG,X_.n_cols)),
                                                                      tau_y(1),
                                                                      y_hat(sum(X % (permutation_matrix * b.value),1)),
                                                                      likelihood(y_,true),
                                                                      rsq(0)
  {

    permutation_matrix.fill(0.0);
    for(uint i = 0; i < groups.n_elem; i++) {
      permutation_matrix(i,groups[i]) = 1.0;
    }

    add(b);
    add(tau_y);
    add(y_hat);
    add(likelihood);
    add(rsq);
  }

  void update() {
    y_hat.value = sum(X % (permutation_matrix * b.value),1);
    rsq.value = as_scalar(1 - var(y - y_hat.value) / var(y));
    b.dnorm(0.0, 0.0001);
    tau_y.dunif(0,100);
    likelihood.dnorm(y_hat.value,tau_y.value);
  }
};

int main() {

  const uint NR = 100;
  const uint NC = 2;
  const uint J = 3;

  mat X = randn<mat>(NR,NC);
  mat y = randn<mat>(NR,1);

  X.col(0).fill(1.0);

  // create fake groups
  ivec groups(NR);
  for(uint i = 0; i < NR; i++) {
    groups[i] = i % J;
  }

  // shift y's by group sums
  vec group_shift(J);
  for(uint i = 0; i < J; i++) {
    group_shift[i] = (i + 1) * 10;
  }
  cout << "group_shift" << endl << group_shift;

  // do the shift on the data
  for(uint i = 0; i < NR; i++) {
    y[i] += group_shift[ groups[i] ];
  }

  vec coefs;
  solve(coefs, X, y);
  vec err = y - X*coefs;

  TestModel m(y,X,groups,J);
  int iterations = 1e5;
  m.sample(iterations, 1e4, 1e4, 10);
  cout << "lm coefs" << endl << coefs;
  cout << "err sd: " << stddev(err,0) << endl;;
  cout << "err tau: " << pow(stddev(err,0),-2) << endl;

  cout << "b: " << endl << m.b.mean();
  cout << "tau_y: " << m.tau_y.mean() << endl;
  cout << "R^2: " << m.rsq.mean() << endl;
  cout << "samples: " << m.b.history.size() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;
  return 0;
};

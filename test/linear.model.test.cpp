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

  Normal<vec> b;
  Uniform<double> tau_y;
  Deterministic<mat> y_hat;
  Normal<mat> likelihood;

  TestModel(const mat& y_,const mat& X_): y(y_), X(X_),
                                          b(randn<vec>(X_.n_cols)),
                                          tau_y(1),
                                          y_hat(X*b.value),likelihood(y_,true)
  {
    add(b);
    add(tau_y);
    add(y_hat);
    add(likelihood);
  }

  void update() {
    y_hat.value = X*b.value;
  }
  double logp() const {
    return b.logp(0.0, 0.0001) + tau_y.logp(0,100) + likelihood.logp(y_hat.value,tau_y.value);
  }
};

// global rng generators
//CppMCGeneratorT MCMCObject::generator_;

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

  TestModel m(y,X);
  int iterations = 1e5;
  m.sample(iterations, 1e4, 10);

  cout << "lm coefs" << endl << coefs;
  cout << "err sd: " << stddev(err,0) << endl;;
  cout << "err tau: " << pow(stddev(err,0),-2) << endl;
  cout << "b: " << endl << m.b.mean();
  cout << "tau_y: " << m.tau_y.mean() << endl;
  cout << "samples: " << m.b.history.size() << endl;
  return 0;
};

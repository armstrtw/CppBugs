#include <iostream>
#include <vector>
#include <armadillo>
#include <cppbugs/cppbugs.hpp>

using namespace arma;
using namespace cppbugs;
using std::cout;
using std::endl;

class LinearModel: public MCModel {
public:
  const mat& y; // given
  const mat& X; // given

  Normal<vec> b;
  Gamma<double> tau_y;
  Deterministic<mat> y_hat;
  Normal<mat> likelihood;
  Deterministic<double> rsq;

  LinearModel(const mat& y_,const mat& X_): y(y_), X(X_),
                                            b(randn<vec>(X_.n_cols)),
                                            tau_y(1),
                                            y_hat(X*b.value),
                                            likelihood(y_,true),
                                            rsq(0)
  {
    add(b);
    add(tau_y);
    add(y_hat);
    add(likelihood);
    add(rsq);
  }

  void update() {
    y_hat.value = X*b.value;
    rsq.value = as_scalar(1 - var(y - y_hat.value) / var(y));
    b.dnorm(0.0, 0.0001);
    tau_y.dgamma(0.1, 0.1);
    likelihood.dnorm(y_hat.value,tau_y.value);
  }
};

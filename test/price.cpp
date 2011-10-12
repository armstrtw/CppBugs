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
  const mat& age; // given
  const mat& price; // given

  Normal<double> a;
  Normal<double> b;
  Gamma<double> tau;
  Deterministic<mat> y_hat;
  Normal<mat> likelihood;

  TestModel(const mat& age_,const mat& price_): age(age_), price(price_),
                                                a(0.0),b(0.0),tau(0.1),
                                                y_hat(a.value + b.value * age),likelihood(price,true)
  {
    add(a);
    add(b);
    add(tau);
    add(y_hat); y_hat.setSaveHistory(false);
    add(likelihood);
  }

  void update() {
    y_hat.value = a.value + b.value * age;
    a.dnorm(0.0, 0.0001);
    b.dnorm(0.0, 0.0001);
    tau.dgamma(0.1, 0.1);
    likelihood.dnorm(y_hat.value,tau.value);
  }
};

int main() {

  // setup data
  double ageraw[] = {13, 14, 14,12, 9, 15, 10, 14, 9, 14, 13, 12, 9, 10, 15, 11, 15, 11, 7,
                     13, 13, 10, 9, 6, 11, 15, 13, 10, 9, 9, 15, 14, 14, 10, 14, 11, 13, 14, 10};
  double priceraw[] = {2950, 2300, 3900, 2800, 5000, 2999, 3950, 2995, 4500, 2800, 1990, 3500, 5100, 3900, 2900,
                       4950, 2000, 3400, 8999, 4000, 2950, 3250, 3950, 4600, 4500, 1600, 3900, 4200, 6500, 3500, 2999, 2600, 3250, 2500, 2400, 3990, 4600, 450,4700};
  const int NR = 39;
  const mat age(ageraw,39,1);
  mat price(priceraw,39,1);
  price /= 1000.0;

  // fit linear model
  mat icept(NR,1); icept.fill(1);
  const mat X = join_rows(icept,age);
  vec coefs;
  solve(coefs, X, price);
  vec err = price - X*coefs;

  TestModel m(age,price);
  int iterations = 1e5;
  m.sample(iterations, 1e4, 1e4, 5);
  cout << "lm coefs" << endl << coefs;
  cout << "err sd: " << stddev(err,0) << endl;;
  cout << "err tau: " << pow(stddev(err,0),-2) << endl;

  cout << "a: " << m.a.mean() << endl;
  cout << "b: " << m.b.mean() << endl;
  cout << "tau: " << m.tau.mean() << endl;
  cout << "yhat history size: " << m.y_hat.history.size() << endl;
  cout << "samples: " << m.b.history.size() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;
  return 0;
};

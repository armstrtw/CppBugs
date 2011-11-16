#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <armadillo>
#include <boost/random.hpp>
#include <cppbugs/cppbugs.hpp>
#include <cppbugs/mcmc.model.hpp>

using namespace arma;
using namespace cppbugs;
using std::cout;
using std::endl;


class Linear {
  double& a_, b_;
public:
  Linear(Normal<double>& a, Normal<double>& b): a_(a.value), b_(b.value) {}
  double operator()() const {
    return a_ + b_;
  }
};

int main() {
  Normal<double> a(10);
  Normal<double> b(20);
  Normal<double> c(30);
  Normal<double> d(40);
  Linear lin(a,b);
  Deterministic<double,Linear> simple(lin);
  a.dnorm(1.5,100.1);
  b.dnorm(1.5,100.1);
  c.dnorm(a,b);
  d.dnorm(a,c);

  //std::function<void ()> foo = [&a,&b]() { cout << "in the FP: " << a.value + b.value << endl; };
  std::function<void ()> foo = [&]() { cout << "in the FP: " << a.value + b.value << endl; }o;
  foo();

  cout << "a lik: " << a.loglik() << endl;
  cout << "b lik: " << b.loglik() << endl;
  cout << "c lik: " << c.loglik() << endl;
  cout << "d lik: " << d.loglik() << endl;

  cout << "lin: " << lin() << endl;
  cout << "simple value: " << simple.value << endl;
  a.value = 0;
  simple.update();
  cout << "simple value 2: " << simple.value << endl;
  return 0;
};

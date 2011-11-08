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
  Normal<double> a(10);
  Normal<double> b(20);
  a.dnorm(1.5,100.1);
  b.dnorm(1.5,100.1);
  return 0;
};

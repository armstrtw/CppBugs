#include <iostream>
#include <vector>
#include <armadillo>
#include <boost/random.hpp>
#include <cppbugs/cppbugs.hpp>

using namespace arma;
using namespace cppbugs;
using std::cout;
using std::endl;

// global rng generators
CppMCGeneratorT MCMCObject::generator_;

int main() {
  int N = 1;
  int incidence_raw[] = {2,3,4,0,3,1,1,8,2,0,2,2,0,2,0,5,0,0,1,3,0,0,1,8,1,3,0,12,2,0,0,0,1,1,0,2,0,5,3,1,2,1,0,0,1,2,0,0,11,0,0,0,1,1,1,0};
  int size_raw[] = {14,12,9,5,22,18,21,22,16,16,20,10,10,9,6,18,25,24,4,17,17,18,20,16,10,9,5,34,9,6,8,6,22,22,18,22,25,27,22,22,10,8,6,5,21,24,19,23,19,2,3,2,19,15,15,15};
  ivec incidence(incidence_raw,N);
  ivec size(size_raw,N);
  vec p(N); p.fill(.5);

  Binomial<ivec> likelihood(incidence,true);
  
  cout << likelihood.logp(size,p) << endl;

  return 0;
}

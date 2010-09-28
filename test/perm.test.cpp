
#include <iostream>
#include <vector>
#include <armadillo>
#include <boost/random.hpp>
#include <cppbugs/cppbugs.hpp>

using namespace arma;
using namespace cppbugs;
using std::cout;
using std::endl;

int main() {
  double herd_raw[] = {1,1,1,1,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15};

  int N = 56;
  int N_herd = 15;
  mat permutation_matrix(N,N_herd);
  vec herd(herd_raw,N,1); herd-=1;
  vec b_herd = randn<vec>(N_herd);

  permutation_matrix.fill(0.0);
  for(uint i = 0; i < N; i++) {
    permutation_matrix(i,herd[i]) = 1.0;
  }
  cout << permutation_matrix << endl;
  cout << sum(permutation_matrix*b_herd,1) << endl;
  return 0;
}

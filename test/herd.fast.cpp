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

class HerdModel: public MCModel {
  const ivec incidence;
  const ivec size;
  const ivec herd;
  const mat fixed;
  int N, N_herd;
  mat indicator_matrix;

public:
  Normal<vec> b;
  Uniform<double> tau_overdisp;
  Uniform<double> tau_b_herd;
  Deterministic<double> sigma_overdisp;
  Deterministic<double> sigma_b_herd;
  Normal<vec> b_herd;
  Normal<vec> overdisp;
  Deterministic<vec> phi;
  Binomial<ivec> likelihood;


  HerdModel(const ivec& incidence_,const ivec& size_,const ivec& herd_,const mat& fixed_,int N_, int N_herd_):
    incidence(incidence_),size(size_),herd(herd_),
    fixed(fixed_),N(N_),N_herd(N_herd_),indicator_matrix(N,N_herd),
    b(randn<vec>(4)),tau_overdisp(1),tau_b_herd(1),
    sigma_overdisp(1),sigma_b_herd(1),
    b_herd(randn<vec>(N_herd_)),overdisp(randn<vec>(N)),
    phi(randu<vec>(N)),
    likelihood(incidence_,true)
  {
    indicator_matrix.fill(0.0);
    for(uint i = 0; i < herd.n_elem; i++) {
      indicator_matrix(i,herd[i]) = 1.0;
    }
    add(b);
    add(tau_overdisp);
    add(tau_b_herd);
    add(sigma_overdisp);
    add(sigma_b_herd);
    add(b_herd);
    add(overdisp);
    add(phi);
    add(likelihood);
  }

  void update() {
    phi.value = fixed*b.value + indicator_matrix*b_herd.value + overdisp.value;
    phi.value = 1/(1+exp(-phi.value));
    sigma_overdisp.value = 1/sqrt(tau_overdisp.value);
    sigma_b_herd.value = 1/sqrt(tau_b_herd.value);
    b.dnorm(0,0.001);
    tau_overdisp.dunif(0,1000);
    tau_b_herd.dunif(0,100);
    b_herd.dnorm(0, tau_b_herd.value);
    overdisp.dnorm(0,tau_overdisp.value);
    likelihood.dbinom(size,phi.value);
  }
};

int main() {
  int incidence_raw[] = {2,3,4,0,3,1,1,8,2,0,2,2,0,2,0,5,0,0,1,3,0,0,1,8,1,3,0,12,2,0,0,0,1,1,0,2,0,5,3,1,2,1,0,0,1,2,0,0,11,0,0,0,1,1,1,0};
  int size_raw[] = {14,12,9,5,22,18,21,22,16,16,20,10,10,9,6,18,25,24,4,17,17,18,20,16,10,9,5,34,9,6,8,6,22,22,18,22,25,27,22,22,10,8,6,5,21,24,19,23,19,2,3,2,19,15,15,15};
  int herd_raw[] = {1,1,1,1,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15};
  double period2_raw[] = {0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0};
  double period3_raw[] = {0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0};
  double period4_raw[] = {0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1};

  int N = 56;
  int N_herd = 15;

  ivec incidence(incidence_raw,N);
  ivec size(size_raw,N);
  ivec herd(herd_raw,N); herd -= 1;
  vec a(N); a.fill(1);
  vec period2(period2_raw,N);
  vec period3(period3_raw,N);
  vec period4(period4_raw,N);
  //mat fixed = randn<mat>(N,4);
  mat fixed(N,4);
  fixed.col(0).fill(1);
  fixed.col(1) = period2;
  fixed.col(2) = period3;
  fixed.col(3) = period4;

  HerdModel m(incidence,size,herd,fixed,N,N_herd);
  //m.sample(1e6,1e5,1e4,50);
  m.sample(1e6,1e4,1e4,10);

  cout << "samples: " << m.b.history.size() << endl;
  cout << "b: " << endl << m.b.mean() << endl;
  cout << "tau_overdisp: " << m.tau_overdisp.mean() << endl;
  cout << "tau_b_herd: " << m.tau_b_herd.mean() << endl;
  cout << "sigma_overdisp: " << m.sigma_overdisp.mean() << endl;
  cout << "sigma_b_herd: " << m.sigma_b_herd.mean() << endl;
  cout << "b_herd: " << endl << m.b_herd.mean() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;
  // cout << "overdisp" << m.overdisp.mean() << endl;
  // cout << "phi" << m.phi.mean() << endl;
  // cout << "phi2" << m.phi2.mean() << endl;
  //cout << "likelihood" << m.likelihood.mean() << endl;

  return 0;
}

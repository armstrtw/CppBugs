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
  int incidence_raw[] = {2,3,4,0,3,1,1,8,2,0,2,2,0,2,0,5,0,0,1,3,0,0,1,8,1,3,0,12,2,0,0,0,1,1,0,2,0,5,3,1,2,1,0,0,1,2,0,0,11,0,0,0,1,1,1,0};
  int size_raw[] = {14,12,9,5,22,18,21,22,16,16,20,10,10,9,6,18,25,24,4,17,17,18,20,16,10,9,5,34,9,6,8,6,22,22,18,22,25,27,22,22,10,8,6,5,21,24,19,23,19,2,3,2,19,15,15,15};
  int herd_raw[] = {1,1,1,1,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15};
  double period2_raw[] = {0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0};
  double period3_raw[] = {0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0};
  double period4_raw[] = {0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1};

  int N = 56;
  int N_herd = 15;

  const ivec incidence(incidence_raw,N);
  const ivec size(size_raw,N);
  ivec herd(herd_raw,N); herd -= 1;
  const vec period2(period2_raw,N);
  const vec period3(period3_raw,N);
  const vec period4(period4_raw,N);

  mat indicator_matrix(N,N_herd);
  indicator_matrix.fill(0.0);
  for(uint i = 0; i < herd.n_elem; i++) {
    indicator_matrix(i,herd[i]) = 1.0;
  }

  mat fixed(N,4);
  fixed.col(0).fill(1);
  fixed.col(1) = period2;
  fixed.col(2) = period3;
  fixed.col(3) = period4;

  vec b(randn<vec>(4));
  vec b_herd(randn<vec>(N_herd));
  vec overdisp(randn<vec>(N));
  vec phi;
  double tau_overdisp(1), tau_b_herd(1), sigma_overdisp(1), sigma_b_herd(1);

  std::function<void ()> model = [&]() {
    phi = fixed*b + indicator_matrix*b_herd + overdisp;
    phi = 1/(1+exp(-phi));
    sigma_overdisp = 1/sqrt(tau_overdisp);
    sigma_b_herd = 1/sqrt(tau_b_herd);
  };

  MCModel m(model);
  m.normal(b).dnorm(0,0.001);
  m.uniform(tau_overdisp).dunif(0,1000);
  m.uniform(tau_b_herd).dunif(0,100);
  m.normal(b_herd).dnorm(0, tau_b_herd);
  m.normal(overdisp).dnorm(0,tau_overdisp);
  m.binomial(incidence).dbinom(size,phi);
  m.deterministic(sigma_overdisp);
  m.deterministic(sigma_b_herd);
  m.deterministic(phi);
  m.sample(1e6,1e5,1e4,50);

  cout << "samples: " << m.getNode(b).history.size() << endl;
  cout << "b: " << endl << m.getNode(b).mean() << endl;
  cout << "tau_overdisp: " << m.getNode(tau_overdisp).mean() << endl;
  cout << "tau_b_herd: " << m.getNode(tau_b_herd).mean() << endl;
  cout << "sigma_overdisp: " << m.getNode(sigma_overdisp).mean() << endl;
  cout << "sigma_b_herd: " << m.getNode(sigma_b_herd).mean() << endl;
  cout << "b_herd: " << endl << m.getNode(b_herd).mean() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;
  //cout << "overdisp" << endl << m.getNode(overdisp).mean() << endl;
  //cout << "phi" << endl << m.getNode(phi).mean() << endl;

  return 0;
}

/*
model {
 for(i in 1:n){
   incidence[i] ~ dbin(phi2[i], size[i])
   logit(phi[i]) <- B.0 + B.period2*period2[i] + B.period3*period3[i] + B.period4*period4[i] + b.herd[herd[i]] + overdisp[i]
   phi2[i] <- max(0.00001, min(phi[i], 0.9999999))
   overdisp[i] ~ dnorm(0,tau.overdisp)
 }
 B.0 ~ dnorm(0,0.001)
 B.period2 ~ dnorm(0, 0.001)
 B.period3 ~ dnorm(0, 0.001)
 B.period4 ~ dnorm(0, 0.001)

 tau.overdisp <- pow(sigma.overdisp, -2)
 sigma.overdisp ~ dunif(0, 1000)
 for(j in 1:n.herd){
   b.herd[j] ~ dnorm(0, tau.b.herd)
 }
 tau.b.herd <- pow(sigma.b.herd, -2)
 sigma.b.herd ~ dunif(0, 100)
}
*/

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
  for(unsigned int i = 0; i < herd.n_elem; i++) {
    indicator_matrix(i,herd[i]) = 1.0;
  }


  double b0(0), b_period2(0), b_period3(0), b_period4(0), tau_overdisp(1), tau_b_herd(1);
  double sigma_overdisp, sigma_b_herd;
  vec b_herd(randn<vec>(N_herd));
  vec overdisp(randn<vec>(N));
  vec phi(N);

  std::function<void ()> model = [&]() {
    phi = b0 + b_period2*period2 + b_period3*period3 + b_period4*period4 + indicator_matrix*b_herd + overdisp;
    phi = 1/(1+exp(-phi));
    sigma_overdisp = 1/sqrt(tau_overdisp);
    sigma_b_herd = 1/sqrt(tau_b_herd);
  };

  MCModel m(model);
  m.normal(b0).dnorm(0,0.001);
  m.normal(b_period2).dnorm(0,0.001);
  m.normal(b_period3).dnorm(0,0.001);
  m.normal(b_period4).dnorm(0,0.001);
  m.uniform(tau_overdisp).dunif(0,1000);
  m.uniform(tau_b_herd).dunif(0,100);
  m.normal(b_herd).dnorm(0, tau_b_herd);
  m.normal(overdisp).dnorm(0,tau_overdisp);
  m.binomial(incidence).dbinom(size,phi);
  m.deterministic(sigma_overdisp);
  m.deterministic(sigma_b_herd);

  m.sample(1e6,5e5,1e4,50);
  //m.sample(1e5,1e4,10);

  cout << "samples: " << m.getNode(b0).history.size() << endl;
  cout << "b0: " << m.getNode(b0).mean() << endl;
  cout << "b_period2: " << m.getNode(b_period2).mean() << endl;
  cout << "b_period3: " << m.getNode(b_period3).mean() << endl;
  cout << "b_period4: " << m.getNode(b_period4).mean() << endl;
  cout << "tau_overdisp: " << m.getNode(tau_overdisp).mean() << endl;
  cout << "tau_b_herd: " << m.getNode(tau_b_herd).mean() << endl;
  cout << "sigma_overdisp: " << m.getNode(sigma_overdisp).mean() << endl;
  cout << "sigma_b_herd: " << m.getNode(sigma_b_herd).mean() << endl;
  cout << "b_herd: " << endl << m.getNode(b_herd).mean() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;
  // cout << "overdisp" << m.overdisp.mean() << endl;
  // cout << "phi" << m.phi.mean() << endl;
  // cout << "phi2" << m.phi2.mean() << endl;
  //cout << "likelihood" << m.likelihood.mean() << endl;

  return 0;
}

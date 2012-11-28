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
  unsigned int herd_raw[] = {1,1,1,1,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15};
  double p1_raw[] = {0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0};
  double p2_raw[] = {0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0};
  double p3_raw[] = {0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1};

  int N = 56;
  int N_herd = 15;

  const ivec incidence(incidence_raw,N);
  const ivec size(size_raw,N);
  uvec herd(herd_raw,N); herd -= 1;
  const vec p1(p1_raw,N);
  const vec p2(p2_raw,N);
  const vec p3(p3_raw,N);

  double b0(0), b1(0), b2(0), b3(0), tau_overdisp(1), tau_b_herd(1);
  double sigma_overdisp, sigma_b_herd;
  vec b_herd(randn<vec>(N_herd));
  vec overdisp(randn<vec>(N));
  vec phi(N);

  std::function<void ()> model = [&]() {
    phi = b0 + b1*p1 + b2*p2 + b3*p3 + b_herd.elem(herd) + overdisp;
    phi = 1/(1+exp(-phi));
    sigma_overdisp = 1/sqrt(tau_overdisp);
    sigma_b_herd = 1/sqrt(tau_b_herd);
  };

  MCModel<boost::minstd_rand> m(model);
  m.link<Normal>(b0, 0, 0.001);
  m.link<Normal>(b1, 0, 0.001);
  m.link<Normal>(b2, 0, 0.001);
  m.link<Normal>(b3, 0, 0.001);
  m.link<Uniform>(tau_overdisp, 0, 1000);
  m.link<Uniform>(tau_b_herd, 0, 100);
  m.link<Normal>(b_herd,0, tau_b_herd);
  m.link<Normal>(overdisp, 0, tau_overdisp);
  m.link<ObservedBinomial>(incidence,size, phi);
  m.link<Deterministic>(phi);
  m.link<Deterministic>(sigma_overdisp);
  m.link<Deterministic>(sigma_b_herd);

  // things to track
  std::vector<double>& b0_hist = m.track<std::vector>(b0);
  std::vector<double>& b1_hist = m.track<std::vector>(b1);
  std::vector<double>& b2_hist = m.track<std::vector>(b2);
  std::vector<double>& b3_hist = m.track<std::vector>(b3);

  std::vector<vec>& b_herd_hist = m.track<std::vector>(b_herd);
  std::vector<vec>& overdisp_hist = m.track<std::vector>(overdisp);

  m.tune(1e4,100);
  m.tune_global(1e4,100);
  m.burn(5e5);
  m.sample(1e6, 50);

  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;
  cout << "samples: " << b0_hist.size() << endl;
  cout << "b0: " << mean(b0_hist.begin(),b0_hist.end()) << endl;
  cout << "b1: " << mean(b1_hist.begin(),b1_hist.end()) << endl;
  cout << "b2: " << mean(b2_hist.begin(),b2_hist.end()) << endl;
  cout << "b3: " << mean(b3_hist.begin(),b3_hist.end()) << endl;
  cout << "b_herd: " << endl << mean(b_herd_hist.begin(),b_herd_hist.end()) << endl;
  cout << "overdisp: " << endl << mean(overdisp_hist.begin(),overdisp_hist.end()) << endl;

  return 0;
}

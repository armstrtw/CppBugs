#include <iostream>
#include <vector>
#include <armadillo>
#include <cppbugs/cppbugs.hpp>
#include <algorithm>

using namespace arma;
using namespace cppbugs;
using std::cout;
using std::endl;

class TestModel: public MCModel {
public:
  const ivec& y; // given
  const mat& X; // given
  const ivec& size; // given

  Normal<vec> b;
  Deterministic<vec> y_hat;
  Deterministic<vec> p_hat;
  Binomial<ivec> likelihood;
  Deterministic<double> rsq;

  TestModel(const ivec& y_,const mat& X_, const ivec& size_, const mat& b_init): y(y_), X(X_), size(size_),
                                          b(b_init),
                                          y_hat(size/(1+exp(-X*b.value))),
                                          p_hat(1/(1+exp(-X*b.value))),
                                          likelihood(y_,true),
                                          rsq(0)
  {
    add(b);
    add(y_hat);
    add(likelihood);
    add(rsq);
  }

  void update() {
    p_hat.value = 1/(1+exp(-X*b.value));
    y_hat.value = p_hat.value % size;
    rsq.value = as_scalar(1 - var(y - y_hat.value) / var(y));
    b.dnorm(0.0, 0.0001);
    likelihood.dbinom(size,p_hat.value);
  }
};

int main() {
  const int NR = 1e2;
  const int NC = 2;
  const ivec size = 1000 * ones<ivec>(NR);
  const mat real_b = mat("0.1; 1.0");
  mat X = mat(NR,NC);
  X.col(0).fill(1);
  X.col(1) = randn<mat>(NR,1);
  ivec y = conv_to<ivec>::from(size / (1+exp(-X*real_b)));


  TestModel m(y,X, size, randn<vec>(NC));
  int iterations = 1e5;
  m.sample(iterations, 1e4, 1e4, 10);
  typename std::list<double>::iterator max_logp_it = std::max_element(m.likelihood.logp_history.begin(), m.likelihood.logp_history.end());
  typename std::list<double>::iterator min_logp_it = std::min_element(m.likelihood.logp_history.begin(), m.likelihood.logp_history.end());
  typename std::list<vec>::iterator max_b_it = m.b.history.begin(); std::advance(max_b_it, std::distance(m.likelihood.logp_history.begin(),max_logp_it));
  typename std::list<vec>::iterator min_b_it = m.b.history.begin(); std::advance(min_b_it, std::distance(m.likelihood.logp_history.begin(),min_logp_it));
  cout << "data generated from b:" << endl << real_b;
  cout << "highest log likelihood attained in sampling:" << *max_logp_it << endl;
  cout << "highest likelihood sampled b:"<<endl<< *max_b_it;
  cout << "lowest log likelihood attained in sampling:" << *min_logp_it << endl;
  cout << "lowest likelihood sampled b:"<<endl<< *min_b_it;


  cout << "mean model likelihood:" << m.likelihood.meanLogLikelihood() << endl;
  cout << "b: " << endl << m.b.mean();
  cout << "R^2: " << m.rsq.mean() << endl;
  cout << "samples: " << m.b.history.size() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;
  return 0;
};

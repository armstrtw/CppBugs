#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <armadillo>
#include <boost/random.hpp>
#include <boost/algorithm/string.hpp>
#include <cppbugs/cppbugs.hpp>

using namespace arma;
using namespace cppbugs;
using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::ifstream;

/*
# Bugs code for multilevel model for radon
# with bsmt as an individual predictor
# varying-intercept model
model {
  for (i in 1:n){
    y[i] ~ dnorm (y.hat[i], tau.y)
    y.hat[i] <- a[county[i]] + b*x[i]
  }
  b ~ dnorm (0, .0001)
  tau.y <- pow(sigma.y, -2)
  sigma.y ~ dunif (0, 100)

  for (j in 1:J){
    a[j] ~ dnorm (mu.a, tau.a)
  }
  mu.a ~ dnorm (0, .0001)
  tau.a <- pow(sigma.a, -2)
  sigma.a ~ dunif (0, 100)
}
*/

class RadonVaryingInterceptModel: public MCModel {
public:
  const vec& level;
  const vec& basement;
  const mat& group;
  int J;

  Normal<vec> a;
  Normal<double> b;
  Deterministic<double> tau_y;
  Uniform<double> sigma_y;
  Normal<double> mu_a;
  Deterministic<double> tau_a;
  Uniform<double> sigma_a;
  Deterministic<mat> y_hat;
  Normal<mat> likelihood;

  RadonVaryingInterceptModel(const vec& level_, const vec& basement_, const mat& group_):
    level(level_),basement(basement_),group(group_),J(group_.n_cols),
    a(randn<vec>(group_.n_cols)), b(0),
    tau_y(1),sigma_y(1),mu_a(0),tau_a(1),sigma_a(1),
    y_hat(randn<mat>(level_.n_rows,1)),likelihood(level_,true)
  {
    add(a);
    add(b);
    add(tau_y);
    add(sigma_y);
    add(mu_a);
    add(tau_a);
    add(sigma_a);
    add(y_hat);
    add(likelihood);
  }

  void update() {
    y_hat.value = group * a.value + b.value * basement;
    tau_y.value = pow(sigma_y.value, -2.0);
    tau_a.value = pow(sigma_a.value, -2.0);
    a.dnorm(mu_a.value, tau_a.value);
    b.dnorm(0, 0.0001);
    sigma_y.dunif(0, 100);
    mu_a.dnorm(0, 0.0001);
    sigma_a.dunif(0, 100);
    likelihood.dnorm(y_hat.value,tau_y.value);
  }
};

void read_csv(string fname, vector< vector<string> >& rows) {
  ifstream fin;
  string buf;
  vector<string> splitbuf;
  typedef vector< boost::iterator_range<string::iterator> > find_vector_type;
  fin.open(fname.c_str());

  if(!fin.is_open()) {
    return;
  }

  while(getline(fin, buf)) {
    boost::split(splitbuf, buf, boost::is_any_of(string(",")));
    rows.push_back(splitbuf);
  }

  fin.close();
}

mat county_to_groups(vector<string>& county) {
  vector<string> unique_counties(county);
  vector<string>::iterator unique_counties_end = unique(unique_counties.begin(), unique_counties.end());
  mat ans(county.size(),std::distance(unique_counties.begin(),unique_counties_end)); ans.fill(0);
  for(size_t i = 0; i < county.size(); i++) {
    ans(i,std::distance(unique_counties.begin(),find(unique_counties.begin(),unique_counties_end,county[i]))) = 1.0;
  }
  return ans;
}

void fixlog(vec& level) {
  uvec ltz = find(level <= 0);
  for(size_t i = 0; i < ltz.n_elem; i++) {
    level[ ltz[i] ] = 0.1;
  }
  level = log(level);
}

int main() {
string file("/home/warmstrong/dvl/scripts/mcmc/radon/srrs.csv");
  vector< vector<string> > rows;
  read_csv(file,rows);

  vec level(rows.size(),1);
  vec basement(rows.size(),1);
  vector<string> county(rows.size());

  for(size_t i = 0; i < rows.size(); i++) {
    county[i] = rows[i][0];
    level[i] = atof(rows[i][1].c_str());
    basement[i] = atof(rows[i][2].c_str());
  }

  fixlog(level);
  mat group(county_to_groups(county));

  RadonVaryingInterceptModel m(level,basement,group);
  m.sample(50e3, 10e3, 1e4, 5);
  cout << "samples: " << m.b.history.size() << endl;
  cout << "a: " << endl << m.a.mean() << endl;
  cout << "b: " << m.b.mean() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;
  return 0;
};


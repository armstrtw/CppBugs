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
#include <cppbugs/deterministics/mcmc.linear.with.const.hpp>
#include <cppbugs/deterministics/mcmc.inv.variance.hpp>

using namespace arma;
using namespace cppbugs;
using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::ifstream;

typedef arma::subview_elem1<double, arma::Mat<uword> > replicatedT;

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

/* old
mat county_to_groups(vector<string>& county) {
  vector<string> unique_counties(county);
  vector<string>::iterator unique_counties_end = unique(unique_counties.begin(), unique_counties.end());
  mat ans(county.size(),std::distance(unique_counties.begin(),unique_counties_end)); ans.fill(0);
  for(size_t i = 0; i < county.size(); i++) {
    ans(i,std::distance(unique_counties.begin(),find(unique_counties.begin(),unique_counties_end,county[i]))) = 1.0;
  }
  return ans;
}
*/

uvec county_to_groups(vector<string>& county) {
  vector<string> unique_counties(county);
  vector<string>::iterator unique_counties_end = unique(unique_counties.begin(), unique_counties.end());
  //uvec ans(county.size(),std::distance(unique_counties.begin(),unique_counties_end)); ans.fill(0);
  uvec ans(county.size()); ans.fill(0);
  for(size_t i = 0; i < county.size(); i++) {
    ans(i) = std::distance(unique_counties.begin(),find(unique_counties.begin(),unique_counties_end,county[i]));
  }
  return ans;
}

void fixlog(vec& level) {
  level.elem(find(level <= 0)).fill(0.1);
  level = log(level);
}

int main() {
  string file("/home/warmstrong/dvl/scripts/mcmc/radon/srrs.csv");
  vector< vector<string> > rows;
  read_csv(file,rows);

  vec level(rows.size(),1);
  const vec& level_const = level;
  vec basement(rows.size(),1);
  vector<string> county(rows.size());

  for(size_t i = 0; i < rows.size(); i++) {
    county[i] = rows[i][0];
    level[i] = atof(rows[i][1].c_str());
    basement[i] = atof(rows[i][2].c_str());
  }

  fixlog(level);
  uvec group(county_to_groups(county));

  vec a(randn<vec>(group.max() + 1));
  double b, tau_y(1), sigma_y(1), mu_a, tau_a(1), sigma_a(1);
  replicatedT a_full = a.elem(group);
  mat y_hat;

  MCModel<boost::minstd_rand> m;

  m.link<Uniform>(sigma_a, 0, 100);
  m.link<InvVariance>(tau_a,sigma_a);
  m.link<Normal>(mu_a, 0, 0.001);
  m.link<Normal>(a, mu_a, tau_a);

  m.link<Normal>(b, 0, 0.001);

  m.link<LinearWithConst>(y_hat,basement,a_full,b);

  m.link<Uniform>(sigma_y, 0, 100);
  m.link<InvVariance>(tau_y,sigma_y);

  m.link<ObservedNormal>(level_const, y_hat, tau_y);

  std::vector<vec>& a_hist = m.track<std::vector>(a);
  std::vector<double>& b_hist = m.track<std::vector>(b);

  m.tune(1e4,100);
  m.tune_global(1e4,100);
  m.burn(1e4);
  m.sample(50e3, 5);

  cout << "samples: " << a_hist.size() << endl;
  cout << "a: " << endl << mean(a_hist.begin(),a_hist.end()) << endl;
  cout << "b: " << mean(b_hist.begin(),b_hist.end()) << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;

  return 0;
};


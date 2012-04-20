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
  vec y_hat;

  std::function<void ()> model = [&]() {
    //y_hat = group * a + b * basement;
    y_hat = a.elem(group) + b * basement;
    tau_y = pow(sigma_y, -2.0);
    tau_a = pow(sigma_a, -2.0);
  };

  MCModel<boost::minstd_rand> m(model);
  m.track<Normal>(a).dnorm(mu_a, tau_a);
  m.track<Normal>(b).dnorm(0, 0.0001);
  m.track<Normal>(mu_a).dnorm(0, 0.0001);
  m.track<Uniform>(sigma_y).dunif(0, 100);
  m.track<Uniform>(sigma_a).dunif(0, 100);
  m.track<ObservedNormal>(level_const).dnorm(y_hat,tau_y);
  m.track<Deterministic>(tau_y);
  m.track<Deterministic>(tau_a);

  m.sample(50e3, 10e3, 1e4, 5);
  cout << "samples: " << m.getNode(b).history.size() << endl;
  cout << "a: " << endl << m.getNode(a).mean() << endl;
  cout << "b: " << m.getNode(b).mean() << endl;
  cout << "acceptance_ratio: " << m.acceptance_ratio() << endl;
  return 0;
};


using namespace arma;
using namespace cppbugs;


const mat X = Rcpp::as<arma::mat>(XR);
const mat y = Rcpp::as<arma::mat>(yr);
int iterations_ = as<int>(iterations);
int burn_ = as<int>(burn);
int adapt_ = as<int>(adapt);
int thin_ = as<int>(thin);

vec b = randn<vec>(X.n_cols);
mat y_hat;
double rsq(0);
double tau_y(1);

std::function<void ()> model = [&]() {
  y_hat = X * b;
  rsq = as_scalar(1 - var(y - y_hat) / var(y));
};

MCModel<boost::minstd_rand> m(model);
m.track<Normal>(b).dnorm(0.0, 0.0001);
m.track<Gamma>(tau_y).dgamma(0.1,0.1);
m.track<ObservedNormal>(y).dnorm(y_hat,tau_y);
m.track<Deterministic>(rsq);

m.sample(iterations_, burn_, adapt_, thin_);

return Rcpp::List::create(Rcpp::Named("b", m.getNode(b).mean()), Rcpp::Named("ar", m.acceptance_ratio()), Rcpp::Named("rsq", m.getNode(rsq).mean()));

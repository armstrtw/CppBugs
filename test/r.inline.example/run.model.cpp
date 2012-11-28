using namespace arma;
using namespace cppbugs;


const mat X = Rcpp::as<arma::mat>(XR);
const mat y = Rcpp::as<arma::mat>(yr);
int iterations_ = as<int>(iterations);
int burn_ = as<int>(burn);
int adapt_ = as<int>(adapt);
int adapt_interval_ = as<int>(adapt_interval);
int thin_ = as<int>(thin);

vec b = randn<vec>(X.n_cols);
mat y_hat = X * b;
double rsq(0);
double tau_y(1);

std::function<void ()> model = [&]() {
  y_hat = X * b;
  rsq = as_scalar(1 - var(y - y_hat) / var(y));
};

MCModel<boost::minstd_rand> m(model);
m.link<Normal>(b, 0.0, 0.0001);
m.link<Gamma>(tau_y, 0.1, 0.1);
m.link<Deterministic>(y_hat);
m.link<ObservedNormal>(y, y_hat, tau_y);
m.link<Deterministic>(rsq);

// things to track
std::vector<vec>& b_hist = m.track<std::vector>(b);
std::vector<double>& rsq_hist = m.track<std::vector>(rsq);


  m.tune(adapt_,adapt_interval_);
  m.tune_global(adapt_,adapt_interval_);
  m.burn(burn_);
  m.sample(iterations_, thin_);
m.sample(, , , );

return Rcpp::List::create(Rcpp::Named("b", mean(b_hist.begin(),b_hist.end())),
                          Rcpp::Named("rsq", mean(rsq_hist.begin(),rsq_hist.end())),
                          Rcpp::Named("ar", m.acceptance_ratio()));
                          


mat X = Rcpp::as<arma::mat>(XR);
mat y = Rcpp::as<arma::mat>(yr);
int iterations_ = as<int>(iterations);
int burn_ = as<int>(burn);
int adapt_ = as<int>(adapt);
int thin_ = as<int>(thin);

LinearModel m(y,X);
m.sample(iterations_, burn_, adapt_, thin_);

return Rcpp::List::create(Rcpp::Named("b", m.b.mean()), Rcpp::Named("ar", m.acceptance_ratio()), Rcpp::Named("rsq", m.rsq.mean()));

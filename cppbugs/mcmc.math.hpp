///////////////////////////////////////////////////////////////////////////
// Copyright (C) 2011 Whit Armstrong                                     //
//                                                                       //
// This program is free software: you can redistribute it and/or modify  //
// it under the terms of the GNU General Public License as published by  //
// the Free Software Foundation, either version 3 of the License, or     //
// (at your option) any later version.                                   //
//                                                                       //
// This program is distributed in the hope that it will be useful,       //
// but WITHOUT ANY WARRANTY; without even the implied warranty of        //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         //
// GNU General Public License for more details.                          //
//                                                                       //
// You should have received a copy of the GNU General Public License     //
// along with this program.  If not, see <http://www.gnu.org/licenses/>. //
///////////////////////////////////////////////////////////////////////////

#pragma once

#include <stdexcept>
#include <armadillo>
#include <cppbugs/mcmc.icsi.log.hpp>
#include <cppbugs/mcmc.arma.extensions.hpp>

// Stochastic/Math related functions
namespace cppbugs {

  static inline double square(double x) {
    return x*x;
  }

  static inline int square(int x) {
    return x*x;
  }

  double cholesky_determinant(const arma::mat& R) {
    return arma::prod(square(R.diag()));
  }

  double mahalanobis(const arma::vec& x, const arma::vec& mu, const arma::mat& sigma) {
    const arma::vec err = x - mu;
    return arma::as_scalar(err.t() * sigma.i() * err);
  }

  double mahalanobis(const arma::rowvec& x, const arma::rowvec& mu, const arma::mat& sigma) {
    const arma::rowvec err = x - mu;
    return arma::as_scalar(err * sigma.i() * err.t());
  }

  double mahalanobis_chol(const arma::rowvec& x, const arma::rowvec& mu, const arma::mat& R) {
    const arma::rowvec err = x - mu;
    const arma::mat Rinv(inv(trimatl(R)));
    return arma::as_scalar(err * Rinv * Rinv.t() * err.t());
  }

  template<typename T, typename U, typename V>
  double normal_logp(const T& x, const U& mu, const V& tau) {
    return arma::accu(0.5*log_approx(0.5*tau/arma::datum::pi) - 0.5 * arma::schur(tau, square(x - mu)));
  }

  template<typename T, typename U, typename V>
  double uniform_logp(const T& x, const U& lower, const V& upper) {
    return (arma::any(arma::vectorise(x < lower)) || arma::any(arma::vectorise(x > upper))) ? -std::numeric_limits<double>::infinity() : -arma::accu(log_approx(upper - lower));
  }

  template<typename T, typename U, typename V>
  double gamma_logp(const T& x, const U& alpha, const V& beta) {
    return arma::any(arma::vectorise(x < 0)) ?
      -std::numeric_limits<double>::infinity() :
      arma::accu(arma::schur((alpha - 1.0),log_approx(x)) - arma::schur(beta,x) - lgamma(alpha) + arma::schur(alpha,log_approx(beta)));
  }

  template<typename T, typename U, typename V>
  double beta_logp(const T& x, const U& alpha, const V& beta) {
    const double one = 1.0;
    return arma::any(arma::vectorise(x <= 0)) || arma::any(arma::vectorise(x >= 1)) || arma::any(arma::vectorise(alpha <= 0)) || arma::any(arma::vectorise(beta <= 0)) ?
      -std::numeric_limits<double>::infinity() :
      arma::accu(lgamma(alpha+beta) - lgamma(alpha) - lgamma(beta) + arma::schur((alpha-one),log_approx(x)) + arma::schur((beta-one),log_approx(one-x)));
  }

  double categorical_logp(const arma::ivec& x, const arma::mat& p) {
    if(arma::any(arma::vectorise(p <= 0)) || arma::any(arma::vectorise(p >= 1)) || arma::any(arma::vectorise(x < 0)) || arma::any(arma::vectorise(x >= p.n_cols))) {
      return -std::numeric_limits<double>::infinity();
    }
    // replace w/ call to p.elems later
    double ans(0);
    for(unsigned int i = 0; i < x.n_rows; i++) {
      ans += log_approx(p(i,x[i]));
    }
    return ans;
  }

  double categorical_logp(const arma::ivec& x, const arma::vec& p) {
    if(arma::any(arma::vectorise(p <= 0)) || arma::any(arma::vectorise(p >= 1)) || arma::any(arma::vectorise(x < 0)) || arma::any(arma::vectorise(x >= p.n_elem))) {
      return -std::numeric_limits<double>::infinity();
    }
    // replace w/ call to p.elems later
    double ans(0);
    for(unsigned int i = 0; i < x.n_rows; i++) {
      ans += log_approx(p(x[i]));
    }
    return ans;
  }

  double categorical_logp(const int x, const arma::vec& p) {
    return log_approx(p[x]);
  }

  template<typename T, typename U, typename V>
  double binomial_logp(const T& x, const U& n, const V& p) {
    if(arma::any(arma::vectorise(p <= 0)) || arma::any(arma::vectorise(p >= 1)) || arma::any(arma::vectorise(x < 0))  || arma::any(arma::vectorise(x > n))) {
      return -std::numeric_limits<double>::infinity();
    }
    return arma::accu(arma::schur(x,log_approx(p)) + arma::schur((n-x),log_approx(1-p)) + arma::factln(n) - arma::factln(x) - arma::factln(n-x));
  }

  template<typename T, typename U>
  double bernoulli_logp(const T& x, const U& p) {
    if( arma::any(arma::vectorise(p <= 0)) || arma::any(arma::vectorise(p >= 1)) || arma::any(arma::vectorise(x < 0))  || arma::any(arma::vectorise(x > 1)) ) {
      return -std::numeric_limits<double>::infinity();
    } else {
      return arma::accu(arma::schur(x,log_approx(p)) + arma::schur((1-x), log_approx(1-p)));
    }
  }

  template<typename T, typename U>
  double poisson_logp(const T& x, const U& mu) {
    if( arma::any(arma::vectorise(mu < 0)) || arma::any(arma::vectorise(x < 0))) {
      return -std::numeric_limits<double>::infinity();
    } else {
      return arma::accu(schur(x,log_approx(mu)) - mu - factln(x));
    }
  }

  template<typename T, typename U>
  double exponential_logp(const T& x, const U& lambda) {
    if(!arma::all(arma::vectorise(x > 0)) || !arma::all(arma::vectorise(lambda > 0)))
      return -std::numeric_limits<double>::infinity();
    return arma::accu(log_approx(lambda) - arma::schur(lambda, x));
  }

  template<typename T, typename U>
  double multivariate_normal_chol_logp(const T& x, const U& mu, const arma::mat& R) {
    static double log_2pi = log(2 * arma::datum::pi);
    double ldet = log(cholesky_determinant(R));
    return -0.5 * (x.n_elem * log_2pi + ldet + mahalanobis_chol(x,mu,R));
  }

  // sigma denotes cov matrix rather than precision matrix
  template<typename T, typename U>
  double multivariate_normal_sigma_logp(const T& x, const U& mu, const arma::mat& sigma) {
    arma::mat R;
    bool chol_succeeded = chol(R,sigma);
    if(!chol_succeeded) { return -std::numeric_limits<double>::infinity(); }

    return multivariate_normal_chol_logp(x, mu, R);
  }

  // sigma denotes cov matrix rather than precision matrix
  double multivariate_normal_sigma_logp(const arma::mat& x, const arma::vec& mu, const arma::mat& sigma) {
    arma::mat R;
    bool chol_succeeded = chol(R,sigma);
    if(!chol_succeeded) { return -std::numeric_limits<double>::infinity(); }
    const arma::rowvec mu_r = mu.t();
    double ans(0);
    for(size_t i = 0; i < x.n_rows; i++) {
      ans += multivariate_normal_chol_logp(x.row(i), mu_r, R);
    }
    return ans;
  }

  double multivariate_normal_chol_logp(const arma::mat& x, const arma::vec& mu, const arma::mat& R) {
    const arma::rowvec mu_r = mu.t();
    double ans(0);
    for(size_t i = 0; i < x.n_rows; i++) {
      ans += multivariate_normal_chol_logp(x.row(i), mu_r, R);
    }
    return ans;
  }

  double wishart_logp(const arma::mat& X, const arma::mat& tau, const unsigned int n) {
    if(X.n_cols != X.n_rows || tau.n_cols != tau.n_rows || X.n_cols != tau.n_rows || X.n_cols > n) { return -std::numeric_limits<double>::infinity(); }
    const double lg2 = log(2.0);
    const int k = X.n_cols;
    const double dx(arma::det(X));
    const double db(arma::det(tau));
    if(dx <= 0 || db <= 0) { return -std::numeric_limits<double>::infinity(); }

    const double ldx(log(dx));
    const double ldb(log(db));
    const arma::mat bx(X * tau);
    const double tbx = arma::trace(bx);

    double cum_lgamma(0);
    for(size_t i = 0; i < X.n_rows; ++i) {
      cum_lgamma += lgamma((n + 1)/2.0);
    }
    return (n - k - 1)/2 * ldx + (n/2.0)*ldb - 0.5*tbx - (n*k/2.0)*lg2 - cum_lgamma;
  }

  double mvcar_logp(const arma::mat& X, const arma::vec& adj, const arma::vec& weight, const arma::vec& numNeigh, const arma::mat& tau) {
    return 0;
  }

} // namespace cppbugs

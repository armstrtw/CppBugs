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

#ifndef MCMC_MATH_HPP
#define MCMC_MATH_HPP

#include <cmath>
#include <armadillo>
#include <boost/math/special_functions/gamma.hpp>

namespace cppbugs {
  // Stochastic/Math related functions

  template<typename U>
  double accu(const U&  x) {
    return arma::accu(x);
  }

  double accu(const double x) {
    return x;
  }

  double dim_size(const double x) {
    return 1;
  }

  double dim_size(const int x) {
    return 1;
  }

  double dim_size(const bool x) {
    return 1;
  }

  template<typename T>
  double dim_size(const T& x) {
    return x.n_elem;
  }

  bool any(const bool x) {
    return x;
  }

  bool any(const arma::umat& x) {
    const arma::umat ans(arma::find(x,1));
    return ans.n_elem > 0;
  }

  double log_gamma(const double x) {
    return boost::math::lgamma(x);
  }

  double factln_single(int n) {
    if(n > 100) {
      return log_gamma(static_cast<double>(n) + 1);
    }
    double ans(1);
    for (int i=n; i>1; i--) {
      ans *= i;
    }
    return log(ans);
  }

  double factln(const int i) {
    static std::vector<double> factln_table;

    if(i < 0) {
      return -std::numeric_limits<double>::infinity();
    }

    if(factln_table.size() < static_cast<size_t>(i+1)) {
      for(int j = factln_table.size(); j < (i+1); j++) {
        factln_table.push_back(factln_single(j));
      }
    }
    //return factln_table.at(i);
    return factln_table[i];
  }

  arma::mat factln(const arma::imat& x) {
    arma::mat ans; ans.copy_size(x);
    for(size_t i = 0; i < x.n_elem; i++) {
      ans[i] = factln(x[i]);
    }
    return ans;
  }

  // basic
  const double schur(const double x, const double y) { return x * y; }
  const double schur(const int x, const double y) { return x * y; }
  const double schur(const double x, const int  y) { return x * y; }

  // arma
  const arma::mat schur(const arma::mat& x, const double y) { return x * y; }
  const arma::mat schur(const double x, const arma::mat& y) { return x * y; }
  const arma::mat schur(const arma::mat& x, const arma::mat& y) { return x % y; }
  
  template<typename T, typename U>
  const arma::mat schur(const arma::Mat<T>& x, const arma::Mat<U>& y) { return x % y; }

  template<typename T, typename U, typename V>
  double normal_logp(const T& x, const U& mu, const V& tau) {
    return accu(0.5*log(0.5*tau/arma::math::pi()) - 0.5 * schur(tau, pow(x - mu,2.0)));
  }

  template<typename T, typename U, typename V>
  double uniform_logp(const T& x, const U& lower, const V& upper) {
    return (any(x < lower) || any(x > upper)) ? -std::numeric_limits<double>::infinity() : -accu(log(upper - lower));
  }

  template<typename T, typename U, typename V>
  double gamma_logp(const T& x, const U& alpha, const V& beta) {
    return any(x < 0 ) ?
      -std::numeric_limits<double>::infinity() :
      accu(schur((alpha - 1.0),log(x)) - schur(beta,x) - log_gamma(alpha) + schur(alpha,log(beta)));
  }

  /*
  double gamma_logp(mat& x, const double alpha, const double beta) {
    return any(x < 0 ) ?
      -std::numeric_limits<double>::infinity() :
      (alpha - 1.0) * log(x) - beta * x - log_gamma(alpha) + alpha *log(beta);
  }
  */

  double binom_logp(const arma::ivec& x, const arma::ivec& n, const arma::vec& p) {
    if(any(p <= 0) || any(p >= 1) || any(x < 0)  || any(x > n)) {
      return -std::numeric_limits<double>::infinity();
    } else {
      //return accu(schur(x,log(p)) + schur((n-x),log(1-p)) + factln(n) - factln(x) - factln(n-x));
      return accu(x % log(p) + (n-x) % log(1-p) + factln(n) - factln(x) - factln(n-x));
    }
  }

  double binom_logp(const int x, const int n, const double p) {
    if(any(p <= 0) || any(p >= 1) || any(x < 0)  || any(x > n)) {
      return -std::numeric_limits<double>::infinity();
    } else {
      return accu(x * log(p) + (n-x) * log(1-p) + factln(n) - factln(x) - factln(n-x));
    }
  }

  template<typename T, typename U>
  double bernoulli_logp(const T& x, const U& p) {

    if( any(p <= 0 ) || any(p >= 1) || any(x < 0)  || any(x > 1) ) {
      return -std::numeric_limits<double>::infinity();
    } else {
      return accu(schur(x,log(p)) + schur((1-x), log(1-p)));
    }
  }
} // namespace cppbugs
#endif // MCMC_STOCHASTIC_HPP

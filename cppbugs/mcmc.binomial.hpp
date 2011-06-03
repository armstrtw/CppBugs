///////////////////////////////////////////////////////////////////////////
// Copyright (C) 2010 Whit Armstrong                                     //
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

#ifndef MCMC_BINOMIAL_HPP
#define MCMC_BINOMIAL_HPP


#include <cmath>
#include <armadillo>
#include <boost/math/special_functions/gamma.hpp>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {

  template<typename T>
  class Binomial : public Stochastic<T> {
  public:
    Binomial(const T& value, const bool observed=false): Stochastic<T>(value,observed) {}

    template<typename U, typename V>
    void dbinom(const U& n, const V& p) {
      arma::uvec p_less_than_zero = find(p <= 0,1);
      arma::uvec p_greater_than_1 = find(p >= 1,1);
      arma::uvec value_less_than_zero = find(Stochastic<T>::value < 0,1);
      arma::uvec value_greater_than_n = find(Stochastic<T>::value > n,1);

      if(p_less_than_zero.n_elem || p_greater_than_1.n_elem || value_less_than_zero.n_elem || value_greater_than_n.n_elem) {
        Stochastic<T>::logp_ = -std::numeric_limits<double>::infinity();
      } else {
	Stochastic<T>::logp_ = accu(Stochastic<T>::value % log(p) + (n-Stochastic<T>::value) % log(1-p) + factln(n)-factln(Stochastic<T>::value)-factln(n-Stochastic<T>::value));
      }
    }
  };

  template<>
  class Binomial <arma::ivec> : public Stochastic<arma::ivec> {
  public:
    Binomial(const arma::ivec& value, const bool observed=false): Stochastic<arma::ivec>(value,observed) {}

    void dbinom(const arma::ivec& n, const arma::mat& p) {
      arma::uvec p_less_than_zero = find(p <= 0,1);
      arma::uvec p_greater_than_1 = find(p >= 1,1);
      arma::uvec value_less_than_zero = find(Stochastic<arma::ivec>::value < 0,1);
      arma::uvec value_greater_than_n = find(Stochastic<arma::ivec>::value > n,1);

      if(p_less_than_zero.n_elem || p_greater_than_1.n_elem || value_less_than_zero.n_elem || value_greater_than_n.n_elem) {
        Stochastic<arma::ivec>::logp_ = -std::numeric_limits<double>::infinity();
      } else {
	Stochastic<arma::ivec>::logp_ = accu(Stochastic<arma::ivec>::value % log(p) + (n-Stochastic<arma::ivec>::value) % log(1-p) + factln(n)-factln(Stochastic<arma::ivec>::value)-factln(n-Stochastic<arma::ivec>::value));
      }
    }
  };
} // namespace cppbugs
#endif // MCMC_BINOMIAL_HPP

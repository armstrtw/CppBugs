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

#ifndef MCMC_BINOMIAL_HPP
#define MCMC_BINOMIAL_HPP


#include <cmath>
#include <armadillo>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {

  template<typename T>
  class Binomial : public DynamicStochastic<T> {
  public:
    Binomial(T& value): DynamicStochastic<T>(value) {}

    template<typename U, typename V>
    Binomial<T>& dbinom(const T n, const arma::mat& p) {
      const T& x = DynamicStochastic<T>::value;
      Stochastic::likelihood_functor = [&x,&n,&p]() {
        return binom_logp(x, n, p);
      };
      return *this;
    }
  };

  template<typename T>
  class ObservedBinomial : public Observed<T> {
  public:
    ObservedBinomial(const T& value): Observed<T>(value) {}

    template<typename U, typename V>
    ObservedBinomial<T>& dbinom(const U& n, const V& p) {
      const T& x = Observed<T>::value;
      Stochastic::likelihood_functor = [&x,&n,&p]() {
        return binom_logp(x, n, p);
      };
      return *this;
    }
  };

} // namespace cppbugs
#endif // MCMC_BINOMIAL_HPP

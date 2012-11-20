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

#include <armadillo>
#include <cppbugs/mcmc.dynamic.stochastic.hpp>
#include <cppbugs/mcmc.observed.hpp>

namespace cppbugs {

  template <typename T,typename U, typename V>
  class Binomial : public DynamicStochastic<T> {
  private:
    const U& n_;
    const V& p_;
  public:
    Binomial(T& value, const U& n, const V& p): DynamicStochastic<T>(value), n_(n), p_(p) { dimension_check(value, n_, p_); }
    const double loglik() const { return binom_logp(DynamicStochastic<T>::value, n_, p_); }
  };

  template <typename T,typename U, typename V>
  class ObservedBinomial : public Observed<T> {
  private:
    const U& n_;
    const V& p_;
  public:
    ObservedBinomial(const T& value, const U& n, const V& p): Observed<T>(value), n_(n), p_(p) { dimension_check(value, n_, p_); }
    const double loglik() const { return binom_logp(Observed<T>::value, n_, p_); }
  };

} // namespace cppbugs
#endif // MCMC_BINOMIAL_HPP

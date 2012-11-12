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

#ifndef MCMC_NORMAL_HPP
#define MCMC_NORMAL_HPP

#include <armadillo>
#include <cppbugs/mcmc.dynamic.stochastic.hpp>
#include <cppbugs/mcmc.observed.hpp>

namespace cppbugs {

  template<typename T, typename U, typename V>
  class Normal : public DynamicStochastic<T> {
  private:
    const U& mu_;
    const V& tau_;
  public:
    Normal(T& value, const U& mu, const V& tau): DynamicStochastic<T>(value), mu_(mu), tau_(tau) { dimension_check(value, mu_, tau_); }
    const double loglik() const { return normal_logp(DynamicStochastic<T>::value,mu_,tau_); }
  };

  template<typename T, typename U, typename V>
  class ObservedNormal : public Observed<T> {
  private:
    const U& mu_;
    const V& tau_;
  public:
    ObservedNormal(const T& value, const U& mu, const V& tau): Observed<T>(value), mu_(mu), tau_(tau) { dimension_check(value, mu_, tau_); }
    const double loglik() const { return normal_logp(Observed<T>::value,mu_,tau_); }
  };

} // namespace cppbugs
#endif // MCMC_NORMAL_HPP

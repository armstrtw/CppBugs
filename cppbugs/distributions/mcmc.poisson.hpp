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

#ifndef MCMC_POISSON_HPP
#define MCMC_POISSON_HPP

#include <armadillo>
#include <cppbugs/mcmc.dynamic.stochastic.hpp>
#include <cppbugs/mcmc.observed.hpp>

namespace cppbugs {

  template <typename T,typename U>
  class Poisson : public DynamicStochastic<T> {
  private:
    const U& mu_;
    const bool destory_mu_;
  public:
    Poisson(T& value, const U& mu): DynamicStochastic<T>(value), mu_(mu), destory_mu_(false) { dimension_check(value, mu_); }
    ~Poisson() {
      if(destory_mu_) { delete &mu_; }
    }
    const double loglik() const { return poisson_logp(DynamicStochastic<T>::value, mu_); }
  };

  template <typename T,typename U>
  class ObservedPoisson : public Observed<T> {
  private:
    const U& mu_;
    const bool destory_mu_;
  public:
    ObservedPoisson(const T& value, const U& mu): Observed<T>(value), mu_(mu), destory_mu_(false) { dimension_check(value, mu_); }
    ~ObservedPoisson() {
      if(destory_mu_) { delete &mu_; }
    }
    const double loglik() const { return poisson_logp(Observed<T>::value, mu_); }
  };

} // namespace cppbugs
#endif // MCMC_POISSON_HPP

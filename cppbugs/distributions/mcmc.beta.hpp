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

#ifndef MCMC_BETA_HPP
#define MCMC_BETA_HPP

#include <armadillo>
#include <cppbugs/mcmc.dynamic.stochastic.hpp>
#include <cppbugs/mcmc.observed.hpp>

namespace cppbugs {

  template<typename T, typename U, typename V>
  class Beta : public DynamicStochastic<T> {
  private:
    const U& alpha_;
    const V& beta_;
  public:
    Beta(T& value, const U& alpha, const V& beta): DynamicStochastic<T>(value), alpha_(alpha), beta_(beta)  { dimension_check(value, alpha_, beta_); }

    // modified jumper to only take jumps on (0,1) interval
    void jump(RngBase& rng) { bounded_jump_impl(rng, DynamicStochastic<T>::value, DynamicStochastic<T>::scale_, 0, 1); }
    const double loglik() const { return beta_logp(DynamicStochastic<T>::value, alpha_, beta_); }
  };

  template<typename T, typename U, typename V>
  class ObservedBeta : public Observed<T> {
  private:
    const U& alpha_;
    const V& beta_;
  public:
    ObservedBeta(const T& value, const U& alpha, const V& beta): Observed<T>(value), alpha_(alpha), beta_(beta)  { dimension_check(value, alpha_, beta_); }
    const double loglik() const { return beta_logp(Observed<T>::value, alpha_, beta_); }
  };

} // namespace cppbugs
#endif // MCMC_BETA_HPP

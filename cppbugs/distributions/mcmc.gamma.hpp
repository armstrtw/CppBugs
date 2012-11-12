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

#ifndef MCMC_GAMMA_HPP
#define MCMC_GAMMA_HPP


#include <cmath>
#include <armadillo>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {

  template<typename T,typename U, typename V>
  class Gamma : public DynamicStochastic<T> {
  private:
    const U& alpha_;
    const V& beta_;
  public:
    Gamma(T& value, const U& alpha, const V& beta): DynamicStochastic<T>(value), alpha_(alpha), beta_(beta) { dimension_check(value, alpha_, beta_); }

    // modified jumper to only take positive jumps
    void jump(RngBase& rng) { positive_jump_impl(rng, DynamicStochastic<T>::value, DynamicStochastic<T>::scale_); }
    const double loglik() const { return gamma_logp(DynamicStochastic<T>::value,alpha_,beta_); }
  };

  template<typename T,typename U, typename V>
  class ObservedGamma : public Observed<T> {
  private:
    const U& alpha_;
    const V& beta_;
  public:
    ObservedGamma(const T& value, const U& alpha, const V& beta): Observed<T>(value), alpha_(alpha), beta_(beta) { dimension_check(value, alpha_, beta_); }
    const double loglik() const { return gamma_logp(Observed<T>::value,alpha_,beta_); }
  };

} // namespace cppbugs
#endif // MCMC_GAMMA_HPP

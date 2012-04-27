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

  template <typename T,typename U, typename V>
  class BetaLikelihiood : public Likelihiood {
    const T& x_;
    const U& alpha_;
    const V& beta_;
  public:
    BetaLikelihiood(  const T& x,  const U& alpha,  const V& beta): x_(x), alpha_(alpha), beta_(beta) {}
    inline double calc() const {
      return beta_logp(x_,alpha_,beta_);
    }
  };

  template<typename T>
  class Beta : public DynamicStochastic<T> {
  public:
    Beta(T& value): DynamicStochastic<T>(value) {}

    // modified jumper to only take jumps on (0,1) interval
    void jump(RngBase& rng) { bounded_jump_impl(rng, DynamicStochastic<T>::value,DynamicStochastic<T>::scale_, 0, 1); }

    template<typename U, typename V>
    Beta<T>& dbeta(const U& alpha, const V& beta) {
      Stochastic::likelihood_functor = new BetaLikelihiood<T,U,V>(DynamicStochastic<T>::value,alpha,beta);
      return *this;
    }
  };

  template<typename T>
  class ObservedBeta : public Observed<T> {
  public:
    ObservedBeta(const T& value): Observed<T>(value) {}

    template<typename U, typename V>
    ObservedBeta<T>& dbeta(const U& alpha, const V& beta) {
      Stochastic::likelihood_functor = new BetaLikelihiood<T,U,V>(Observed<T>::value,alpha,beta);
      return *this;
    }
  };

} // namespace cppbugs
#endif // MCMC_BETA_HPP

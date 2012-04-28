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

  template <typename T,typename U, typename V>
  class GammaLikelihiood : public Likelihiood {
    const T& x_;
    const U& alpha_;
    const V& beta_;
  public:
    GammaLikelihiood(const T& x,  const U& alpha,  const V& beta): x_(x), alpha_(alpha), beta_(beta) { dimension_check(x_, alpha_, beta_); }
    inline double calc() const {
      return gamma_logp(x_,alpha_,beta_);
    }
  };

  template<typename T>
  class Gamma : public DynamicStochastic<T> {
  public:
    Gamma(T& value): DynamicStochastic<T>(value) {}

    // modified jumper to only take positive jumps
    void jump(RngBase& rng) { positive_jump_impl(rng, DynamicStochastic<T>::value,DynamicStochastic<T>::scale_); }

    template<typename U, typename V>
    Gamma<T>& dgamma(const U& alpha, const V& beta) {
      Stochastic::likelihood_functor = new GammaLikelihiood<T,U,V>(DynamicStochastic<T>::value,alpha,beta);
      return *this;
    }
  };

  template<typename T>
  class ObservedGamma : public Observed<T> {
  public:
    ObservedGamma(const T& value): Observed<T>(value) {}

    template<typename U, typename V>
    ObservedGamma<T>& dgamma(const U& alpha, const V& beta) {
      Stochastic::likelihood_functor = new GammaLikelihiood<T,U,V>(Observed<T>::value,alpha,beta);
      return *this;
    }
  };

} // namespace cppbugs
#endif // MCMC_GAMMA_HPP

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

  template <typename T,typename U, typename V>
  class NormalLikelihiood : public Likelihiood {
    const T& x_;
    const U& mu_;
    const V& tau_;
  public:
    NormalLikelihiood(  const T& x,  const U& mu,  const V& tau): x_(x), mu_(mu), tau_(tau) {}
    inline double calc() const {
      return normal_logp(x_,mu_,tau_);
    }
  };

  template<typename T>
  class Normal : public DynamicStochastic<T> {
  public:
    Normal(T& value): DynamicStochastic<T>(value) {}

    template<typename U, typename V>
    Normal<T>& dnorm(const U& mu, const V& tau) {
      Stochastic::likelihood_functor = new NormalLikelihiood<T,U,V>(DynamicStochastic<T>::value,mu,tau);
      return *this;
    }
  };

  template<typename T>
  class ObservedNormal : public Observed<T> {
  public:
    ObservedNormal(const T& value): Observed<T>(value) {}

    template<typename U, typename V>
    ObservedNormal<T>& dnorm(const U& mu, const V& tau) {
      Stochastic::likelihood_functor = new NormalLikelihiood<T,U,V>(Observed<T>::value,mu,tau);
      return *this;
    }
  };

} // namespace cppbugs
#endif // MCMC_NORMAL_HPP

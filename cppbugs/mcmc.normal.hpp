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

#include <functional>
#include <cmath>
#include <armadillo>
#include <cppbugs/mcmc.dynamic.stochastic.hpp>
#include <cppbugs/mcmc.observed.hpp>

namespace cppbugs {


  template<typename T>
  class Normal : public DynamicStochastic<T> {
  public:
    Normal(T& value): DynamicStochastic<T>(value) {}

    template<typename U, typename V>
    Normal<T>& dnorm(const U& mu, const V& tau) {
      const T& x = DynamicStochastic<T>::value;
      Stochastic::likelihood_functor = [&x,&mu,&tau]() {
        return normal_logp(x,mu,tau);
      };
      return *this;
    }
  };

  template<typename T>
  class ObservedNormal : public Observed<T> {
  public:
    ObservedNormal(const T& value): Observed<T>(value) {}

    template<typename U, typename V>
    ObservedNormal<T>& dnorm(const U& mu, const V& tau) {
      const T& x = Observed<T>::value;
      Stochastic::likelihood_functor = [&x,&mu,&tau]() {
        return normal_logp(x,mu,tau);
      };
      return *this;
    }
  };


} // namespace cppbugs
#endif // MCMC_NORMAL_HPP

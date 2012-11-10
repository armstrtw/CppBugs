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

#ifndef MCMC_CATEGORICAL_HPP
#define MCMC_CATEGORICAL_HPP

#include <armadillo>
#include <cppbugs/mcmc.dynamic.stochastic.hpp>
#include <cppbugs/mcmc.observed.hpp>

namespace cppbugs {

  template <typename T,typename U>
  class CategoricalLikelihiood : public Likelihiood {
    const T& x_;
    const U& p_;
  public:
    CategoricalLikelihiood(const T& x,  const U& p): x_(x), p_(p) {}
    inline double calc() const {
      return categorical_logp(x_,p_);
    }
  };

  template<typename T>
  class Categorical : public DynamicStochastic<T> {
  public:
    Categorical(T& value): DynamicStochastic<T>(value) {}

    template<typename U>
    Categorical<T>& dcat(const U& p) {
      Stochastic::likelihood_functor = new CategoricalLikelihiood<T,U>(DynamicStochastic<T>::value,p);
      return *this;
    }
  };

  template<typename T>
  class ObservedCategorical : public Observed<T> {
  public:
    ObservedCategorical(const T& value): Observed<T>(value) {}

    template<typename U>
    ObservedCategorical<T>& dcat(const U& p) {
      Stochastic::likelihood_functor = new CategoricalLikelihiood<T,U>(Observed<T>::value, p);
      return *this;
    }
  };

} // namespace cppbugs
#endif // MCMC_CATEGORICAL_HPP

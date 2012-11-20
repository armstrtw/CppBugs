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
  class Categorical : public DynamicStochastic<T> {
  private:
    const U& p_;
  public:
    Categorical(T& value, const U& p): DynamicStochastic<T>(value), p_(p) {}
    const double loglik() const { return categorical_logp(DynamicStochastic<T>::value, p_); }
  };

  template <typename T,typename U>
  class ObservedCategorical : public Observed<T> {
  private:
    const U& p_;
  public:
    ObservedCategorical(const T& value, const U& p): Observed<T>(value), p_(p) {}
    const double loglik() const { return categorical_logp(Observed<T>::value, p_); }
  };

} // namespace cppbugs
#endif // MCMC_CATEGORICAL_HPP

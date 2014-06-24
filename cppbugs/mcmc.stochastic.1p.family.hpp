///////////////////////////////////////////////////////////////////////////
// Copyright (C) 2014 Whit Armstrong                                     //
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

#ifndef MCMC_STOCHASTIC_1P_FAMILY_HPP
#define MCMC_STOCHASTIC_1P_FAMILY_HPP

#include <armadillo>
#include <cppbugs/mcmc.dynamic.stochastic.hpp>
#include <cppbugs/mcmc.observed.hpp>

namespace cppbugs {

  template<typename T, typename U, double LOGLIKFUN(const T&, const U&)>
  class Stochastic1p : public DynamicStochastic<T> {
  private:
    const U& p1_;
    const bool destory_p1_;
  public:
    Stochastic1p(T& value, const U& p1): DynamicStochastic<T>(value), p1_(p1), destory_p1_(false) { dimension_check(value,p1); }
    // special ctors to capture rvalues and convert to heap objects
    Stochastic1p(T& value, const U&& p1): DynamicStochastic<T>(value), p1_(p1), destory_p1_(true) { dimension_check(value, p1_); }

    ~Stochastic1p() {
      if(destory_p1_) { delete &p1_; }
    }
    double loglik() const { return LOGLIKFUN(DynamicStochastic<T>::value,p1_); }
  };

  template<typename T, typename U, double LOGLIKFUN(const T&, const U&)>
  class ObservedStochastic1p : public Observed<T> {
  private:
    const U& p1_;
    const bool destory_p1_;
  public:
    ObservedStochastic1p(const T& value, const U& p1): Observed<T>(value), p1_(p1), destory_p1_(false) { dimension_check(value,p1); }
    ObservedStochastic1p(T& value, const U&& p1): Observed<T>(value), p1_(p1), destory_p1_(true) { dimension_check(value, p1_); }
    ~ObservedStochastic1p() {
      if(destory_p1_) { delete &p1_; }
    }
    double loglik() const { return LOGLIKFUN(Observed<T>::value,p1_); }
  };

} // namespace cppbugs
#endif // MCMC_STOCHASTIC_1P_FAMILY_HPP

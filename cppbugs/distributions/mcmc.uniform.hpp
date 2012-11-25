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

#ifndef MCMC_UNIFORM_HPP
#define MCMC_UNIFORM_HPP


#include <cmath>
#include <armadillo>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {

  template<typename T, typename U, typename V>
  class Uniform : public DynamicStochastic<T> {
  private:
    const U& lower_;
    const V& upper_;
    const bool destory_lower_, destory_upper_;
  public:
    Uniform(T& value, const U& lower, const V& upper): DynamicStochastic<T>(value), lower_(lower), upper_(upper), destory_lower_(false), destory_upper_(false) { dimension_check(value, lower_, upper_); }
    // special ifdef for const bug/feature introduced in gcc 4.7
#if GCC_VERSION > 40700
    Uniform(T& value, const U&& lower, const V& upper): DynamicStochastic<T>(value), lower_(*(new U(lower))), upper_(upper), destory_lower_(true), destory_upper_(false) { dimension_check(value, lower_, upper_); }
    Uniform(T& value, const U& lower, const V&& upper): DynamicStochastic<T>(value), lower_(lower), upper_(*(new V(upper))), destory_lower_(false), destory_upper_(true) { dimension_check(value, lower_, upper_); }
    Uniform(T& value, const U&& lower, const V&& upper): DynamicStochastic<T>(value),lower_(*(new U(lower))), upper_(*(new V(upper))), destory_lower_(true), destory_upper_(true)   { dimension_check(value, lower_, upper_); }
#endif
    ~Uniform() {
      if(destory_lower_) { delete &lower_; }
      if(destory_upper_) { delete &upper_; }
    }
    const double loglik() const { return uniform_logp(DynamicStochastic<T>::value,lower_,upper_); }
  };

  template<typename T, typename U, typename V>
  class ObservedUniform : public Observed<T> {
  private:
    const U& lower_;
    const V& upper_;
    const bool destory_lower_, destory_upper_;
  public:
    ObservedUniform(const T& value, const U& lower, const V& upper): Observed<T>(value), lower_(lower), upper_(upper), destory_lower_(false), destory_upper_(false) { dimension_check(value, lower_, upper_); }
    // special ifdef for const bug/feature introduced in gcc 4.7
#if GCC_VERSION > 40700
    ObservedUniform(T& value, const U&& lower, const V& upper): Observed<T>(value), lower_(*(new U(lower))), upper_(upper), destory_lower_(true), destory_upper_(false) { dimension_check(value, lower_, upper_); }
    ObservedUniform(T& value, const U& lower, const V&& upper): Observed<T>(value), lower_(lower), upper_(*(new V(upper))), destory_lower_(false), destory_upper_(true) { dimension_check(value, lower_, upper_); }
    ObservedUniform(T& value, const U&& lower, const V&& upper): Observed<T>(value),lower_(*(new U(lower))), upper_(*(new V(upper))), destory_lower_(true), destory_upper_(true)   { dimension_check(value, lower_, upper_); }
#endif
    ~ObservedUniform() {
      if(destory_lower_) { delete &lower_; }
      if(destory_upper_) { delete &upper_; }
    }
    const double loglik() const { return uniform_logp(Observed<T>::value,lower_,upper_); }
  };


} // namespace cppbugs
#endif // MCMC_UNIFORM_HPP

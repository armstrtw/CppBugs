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

#ifndef MCMC_STOCHASTIC_2P_FAMILY_HPP
#define MCMC_STOCHASTIC_2P_FAMILY_HPP

#include <armadillo>
#include <cppbugs/mcmc.dynamic.stochastic.hpp>
#include <cppbugs/mcmc.observed.hpp>

namespace cppbugs {

  template<typename T, typename U, typename V, double LOGLIKFUN(const T&, const U&, const V&)>
  class Stochastic2p : public DynamicStochastic<T> {
  private:
    const U& p1_;
    const V& p2_;
    const bool destory_p1_, destory_p2_;
  public:
    Stochastic2p(T& value, const U& p1, const V& p2): DynamicStochastic<T>(value), p1_(p1), p2_(p2), destory_p1_(false), destory_p2_(false) { dimension_check(value, p1_, p2_); }
    // special ctors to capture rvalues and convert to heap objects
    Stochastic2p(T& value, const U&& p1, const V& p2): DynamicStochastic<T>(value), p1_(*(new U(p1))), p2_(p2), destory_p1_(true), destory_p2_(false) { dimension_check(value, p1_, p2_); }
    Stochastic2p(T& value, const U& p1, const V&& p2): DynamicStochastic<T>(value), p1_(p1), p2_(*(new V(p2))), destory_p1_(false), destory_p2_(true) { dimension_check(value, p1_, p2_); }
    Stochastic2p(T& value, const U&& p1, const V&& p2): DynamicStochastic<T>(value),p1_(*(new U(p1))), p2_(*(new V(p2))), destory_p1_(true), destory_p2_(true)   { dimension_check(value, p1_, p2_); }

    ~Stochastic2p() {
      if(destory_p1_) { delete &p1_; }
      if(destory_p2_) { delete &p2_; }
    }
    const double loglik() const { return LOGLIKFUN(DynamicStochastic<T>::value,p1_,p2_); }
  };

  template<typename T, typename U, typename V, double LOGLIKFUN(const T&, const U&, const V&)>
  class ObservedStochastic2p : public Observed<T> {
  private:
    const U& p1_;
    const V& p2_;
    const bool destory_p1_, destory_p2_;
  public:
    ObservedStochastic2p(const T& value, const U& p1, const V& p2): Observed<T>(value), p1_(p1), p2_(p2), destory_p1_(false), destory_p2_(false) { dimension_check(value, p1_, p2_); }
    // special ctors to capture rvalues and convert to heap objects
    ObservedStochastic2p(const T& value, const U&& p1, const V& p2): Observed<T>(value), p1_(*(new U(p1))), p2_(p2), destory_p1_(true), destory_p2_(false) { dimension_check(value, p1_, p2_); }
    ObservedStochastic2p(const T& value, const U& p1, const V&& p2): Observed<T>(value), p1_(p1), p2_(*(new V(p2))), destory_p1_(false), destory_p2_(true) { dimension_check(value, p1_, p2_); }
    ObservedStochastic2p(const T& value, const U&& p1, const V&& p2): Observed<T>(value),p1_(*(new U(p1))), p2_(*(new V(p2))), destory_p1_(true), destory_p2_(true)   { dimension_check(value, p1_, p2_); }

    ~ObservedStochastic2p() {
      if(destory_p1_) { delete &p1_; }
      if(destory_p2_) { delete &p2_; }
    }
    const double loglik() const { return LOGLIKFUN(Observed<T>::value,p1_,p2_); }
  };

} // namespace cppbugs
#endif // MCMC_STOCHASTIC_2P_FAMILY_HPP

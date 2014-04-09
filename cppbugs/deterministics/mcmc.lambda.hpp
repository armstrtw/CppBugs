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

#ifndef MCMC_LAMBDA_HPP
#define MCMC_LAMBDA_HPP

#include <cppbugs/mcmc.dynamic.hpp>
#include <functional>

namespace cppbugs {

  template<typename T, typename U>
  class Lambda1 : public Deterministic<T> {
    const U& a_;
    std::function<T(const U&)> f_;
  public:
    Lambda1(T& value, std::function<const T(const U&)> f, const U& a): Deterministic<T>(value), f_(f), a_(a) {}
    void jump(RngBase& rng) {
      Deterministic<T>::value = f_(a_);
    }
  };

  template<typename T, typename U, typename V>
  class Lambda2 : public Deterministic<T> {
    const U& a_;
    const U& b_;
    std::function<T(const U&,const V&)> f_;
  public:
    Lambda2(T& value, std::function<T(const U&,const V&)> f, const U& a,const V& b): Deterministic<T>(value), f_(f), a_(a), b_(b) {
      Deterministic<T>::value = f_(a_,b_);
    }
    void jump(RngBase& rng) {
      Deterministic<T>::value = f_(a_,b_);
    }
  };

  template<typename T, typename U, typename V, typename W>
  class Lambda3 : public Deterministic<T> {
    const U& a_;
    const V& b_;
    const W& c_;
    std::function<T(const U&,const V&,const W&)> f_;
  public:
    Lambda3(T& value, std::function<T(const U&,const V&,const W&)> f, const U& a,const V& b,const W& c): Deterministic<T>(value), f_(f), a_(a), b_(b), c_(c) {
      Deterministic<T>::value = f_(a_,b_,c_);
    }
    void jump(RngBase& rng) {
      Deterministic<T>::value = f_(a_,b_,c_);
    }
  };

  template<typename T, typename U, typename V, typename W, typename X>
  class Lambda4 : public Deterministic<T> {
    const U& a_;
    const V& b_;
    const W& c_;
    const X& d_;
    std::function<T(const U&,const V&,const W&,const X&)> f_;
  public:
    Lambda4(T& value, std::function<T(const U&,const V&,const W&,const X&)> f, const U& a,const V& b,const W& c, const X& d): Deterministic<T>(value), f_(f), a_(a), b_(b), c_(c), d_(d) {
      Deterministic<T>::value = f_(a_,b_,c_,d_);
    }
    void jump(RngBase& rng) {
      Deterministic<T>::value = f_(a_,b_,c_,d_);
    }
  };

} // namespace cppbugs
#endif //MCMC_LAMBDA_HPP

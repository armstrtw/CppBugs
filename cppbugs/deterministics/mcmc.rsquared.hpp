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

#pragma once

#include <cppbugs/mcmc.dynamic.hpp>

namespace cppbugs {

  template<typename T, typename U, typename V>
  class Rsquared : public Deterministic<T> {
    const U& y_;
    const V& y_hat_;
  public:
    Rsquared(T& x, const U& y, const V& y_hat): Deterministic<T>(x), y_(y), y_hat_(y_hat) {
      Deterministic<T>::value = as_scalar(1 - var(y_ - y_hat_) / var(y_));
    }
    void jump(RngBase& rng) {
      Deterministic<T>::value = as_scalar(1 - var(y_ - y_hat_) / var(y_));
    }
  };
} // namespace cppbugs

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

#include <cppbugs/mcmc.rng.base.hpp>

#ifndef MCMC_JUMP_HPP
#define MCMC_JUMP_HPP

namespace cppbugs {

  // needed for completeness
  void jump_impl(RngBase& rng, int& value, const double scale) {
    value += rng.normal() * scale;
  }

  void jump_impl(RngBase& rng, double& value, const double scale) {
    value += rng.normal() * scale;
  }

  void positive_jump_impl(RngBase& rng, double& value, const double scale) {
    double new_value;
    do {
      new_value = value + rng.normal() * scale;
    } while (new_value < 0);

    value = new_value;
  }

  template<typename T>
  void jump_impl(RngBase& rng, T& value, const double scale) {
    for(size_t i = 0; i < value.n_elem; i++) {
      value[i] += rng.normal() * scale;
    }
  }

  template<typename T>
  void positive_jump_impl(RngBase& rng, T& value, const double scale) {
    for(size_t i = 0; i < value.n_elem; i++) {
      positive_jump_impl(rng, value[i], scale);
    }
  }

} // namespace cppbugs
#endif // MCMC_JUMP_HPP

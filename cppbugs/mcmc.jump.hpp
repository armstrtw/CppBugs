///////////////////////////////////////////////////////////////////////////
// Copyright (C) 2010 Whit Armstrong                                     //
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

#include <cppbugs/mcmc.rng.hpp>

#ifndef MCMC_JUMP_HPP
#define MCMC_JUMP_HPP

namespace cppbugs {

  template<typename T>
  void jump_impl(RngBase& rng, T& value, const double scale) {
    for(size_t i = 0; i < value.n_elem; i++) {
      value[i] += rng.normal() * scale;
    }
  }

  void jump_impl(RngBase& rng, double& value, const double scale) {
    value += rng.normal() * scale;
  }

} // namespace cppbugs
#endif // MCMC_JUMP_HPP

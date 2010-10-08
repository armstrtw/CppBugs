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

#ifndef MCMC_DETERMINISTIC_HPP
#define MCMC_DETERMINISTIC_HPP

#include <cppbugs/mcmc.specialized.hpp>

namespace cppbugs {

  template<typename T>
  class Deterministic : public MCMCSpecialized<T> {
  public:
    Deterministic(const T& value): MCMCSpecialized<T>(value) {}
    bool isDeterministc() const { return true; }
    bool isStochastic() const { return false; }
    bool isObserved() const { return true; }
  };

} // namespace cppbugs
#endif //MCMC_DETERMINISTIC_HPP

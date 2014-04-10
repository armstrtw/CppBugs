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

#ifndef MCMC_DYNAMIC_HPP
#define MCMC_DYNAMIC_HPP

#include <cppbugs/mcmc.specialized.hpp>
#include <cppbugs/mcmc.utils.hpp>

namespace cppbugs {

  template<typename T>
  class Dynamic : public MCMCSpecialized<T> {
  public:
    T& value;
    T old_value;
    Dynamic(T& shape): MCMCSpecialized<T>(), value(shape), old_value(shape) {}

    void preserve() { old_value = value; }
    void revert() { value = old_value; }
    double size() const { return dim_size(value); }
  };

} // namespace cppbugs
#endif //MCMC_DYNAMIC_HPP

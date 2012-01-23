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

#include <list>
#include <cppbugs/mcmc.specialized.hpp>

namespace cppbugs {

  template<typename T>
  class Dynamic : public MCMCSpecialized<T> {
  public:
    T& value;
    T old_value;
    Dynamic(T& shape): MCMCSpecialized<T>(), value(shape), old_value(shape) {}

    static int sum_dims(const double& value) { return 1; }
    static int sum_dims(const arma::mat& value) { return value.n_elem; }
    static int sum_dims(const arma::ivec& value) { return value.n_elem; }

    void preserve() { old_value = value; }
    void revert() { value = old_value; }
    void tally() { if(MCMCSpecialized<T>::save_history_) { MCMCSpecialized<T>::history.push_back(value); } }
  };

} // namespace cppbugs
#endif //MCMC_DYNAMIC_HPP

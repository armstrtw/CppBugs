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

#ifndef MCMC_OBSERVED_HPP
#define MCMC_OBSERVED_HPP

#include <cppbugs/mcmc.specialized.hpp>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {

  template<typename T>
  class Observed : public MCMCSpecialized<T>, public Stochastic {
  public:
    const T& value;
    Observed(const T& shape): MCMCSpecialized<T>(), value(shape) {}

    void jump(RngBase&) {}
    void accept() {}
    void reject() {}
    void tune() {}
    void preserve() {}
    void revert() {}
    void tally() {}
    bool isDeterministc() const { return false; }
    bool isStochastic() const { return true; }
    bool isObserved() const { return true; }
    void setScale(const double) {}
    double getScale() const { return 0; }
    double size() const { return 0; }
  };

} // namespace cppbugs
#endif //MCMC_OBSERVED_HPP

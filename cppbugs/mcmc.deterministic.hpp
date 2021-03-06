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

  template<typename T>
  class Deterministic : public Dynamic<T> {
  public:
    Deterministic(T& value): Dynamic<T>(value) {}
    //void jump(RngBase& rng) {}
    void accept() { throw std::logic_error("Cannot accept a deterministic."); }
    void reject(){ throw std::logic_error("Cannot reject a deterministic."); }
    void tune() { throw std::logic_error("Cannot tune a deterministic."); }
    // in Dynamic: void preserve()
    // in Dynamic: void revert()
    // in Dynamic: void tally()
    bool isDeterministc() const { return true; }
    bool isStochastic() const { return false; }
    bool isObserved() const { return false; }

    //void setScale(const double scale) { throw std::logic_error("Cannot setScale of a deterministic."); }
    //double getScale() const { throw std::logic_error("Cannot getScale of a deterministic."); return 0; }

    void setScale(const double scale) {}
    double getScale() const { return 0; }
  };

} // namespace cppbugs

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

#ifndef MCMC_OBJECT_HPP
#define MCMC_OBJECT_HPP

#include <cppbugs/mcmc.rng.base.hpp>

namespace cppbugs {

  class MCMCObject {
  public:
    MCMCObject() {}
    virtual ~MCMCObject() {}
    virtual void jump(RngBase& rng) = 0;
    virtual void accept() = 0;
    virtual void reject() = 0;
    virtual void tune() = 0;
    virtual void preserve() = 0;
    virtual void revert() = 0;
    virtual void tally() = 0;
    virtual bool isDeterministc() const = 0;
    virtual bool isStochastic() const = 0;
    virtual bool isObserved() const = 0;
    virtual void setScale(const double scale) = 0;
    virtual double getScale() const = 0;
    virtual double size() const = 0;
  };

} // namespace cppbugs
#endif // MCMC_OBJECT_HPP

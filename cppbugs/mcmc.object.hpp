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

#ifndef MCMC_OBJECT_HPP
#define MCMC_OBJECT_HPP

#include <cppbugs/mcmc.model.base.hpp>

namespace cppbugs {

  class MCMCObject {
  public:
    MCMCObject() {}
    virtual void jump(RngBase& rng) {}
    virtual void tune() {}
    virtual const double* getLogp() const { return static_cast<double*>(NULL); }
    virtual void accept() {}
    virtual void reject() {}
    virtual void preserve() = 0;        // in mcmc.specialized
    virtual void revert() = 0;          // in mcmc.specialized
    virtual void tally() = 0; // in mcmc.specialized
    virtual void print() const = 0;     // in mcmc.specialized
    virtual bool isDeterministc() const = 0;
    virtual bool isStochastic() const = 0;
    virtual bool isObserved() const = 0;
    virtual int getSize() const = 0; // in mcmc.specialized
    virtual void setScale(const double scale) = 0;
  };

} // namespace cppbugs
#endif // MCMC_OBJECT_HPP

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

#ifndef MCMC_RNG_BASE_HPP
#define MCMC_RNG_BASE_HPP


namespace cppbugs {

  class RngBase {
  public:
    RngBase() {}
    virtual double normal() = 0;
    virtual double uniform() = 0;
    //virtual int poisson(n) = 0;
    // etc...
  };

} // namespace cppbugs
#endif // MCMC_RNG_BASE_HPP

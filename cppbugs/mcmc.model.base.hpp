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

#ifndef MCMC_MODEL_BASE_HPP
#define MCMC_MODEL_BASE_HPP

#include <cmath>
#include <boost/random.hpp>
#include <cppbugs/mcmc.rng.hpp>

namespace cppbugs {

  class MCModelBase {
  private:
    SpecializedRng<boost::minstd_rand> rng_;
    bool bad_logp(const double value) const { return std::isnan(value) || value == -std::numeric_limits<double>::infinity() ? true : false; }
  public:
    MCModelBase() {}
    virtual void update() = 0;
    bool reject(const double value, const double old_logp) {
      return bad_logp(value) || log(rng_.uniform()) > value - old_logp ? true : false;
    }
  };
} // namespace cppbugs
#endif // MCMC_MODEL_BASE_HPP

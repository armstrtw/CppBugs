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

#ifndef MCMC_UNIFORM_HPP
#define MCMC_UNIFORM_HPP

#include <armadillo>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {

  template<typename T>
  class Uniform : public Stochastic<T> {
  public:
    Uniform(const T& x, const bool observed = false): Stochastic<T>(x,observed) {}
    void logp(const double lower, const double upper) {
      Stochastic<T>::logp_ = (Stochastic<T>::value < lower || Stochastic<T>::value > upper) ? -std::numeric_limits<double>::infinity() : -log(upper - lower);
    }
  };

  template<typename T>
  class UniformStatic : public Stochastic<T> {
    double lower_, upper_;
  public:
    UniformStatic(const T& x, const double lower, const double upper, const bool observed = false): Stochastic<T>(x,observed),lower_(lower),upper_(upper) {}
    void logp() {
      Stochastic<T>::logp_ = (Stochastic<T>::value < lower_ || Stochastic<T>::value > upper_) ? -std::numeric_limits<double>::infinity() : -log(upper_ - lower_);
    }
    void jump(RngBase& rng) {
      const T oldvalue(Stochastic<T>::value);
      do {
        Stochastic<T>::value = oldvalue + rng.normal() * Stochastic<T>::scale_;
      } while(Stochastic<T>::value < lower_ || Stochastic<T>::value > upper_);
    }
  };
} // namespace cppbugs
#endif // MCMC_UNIFORM_HPP

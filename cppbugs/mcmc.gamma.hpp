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

#ifndef MCMC_GAMMA_HPP
#define MCMC_GAMMA_HPP

#include <cmath>
#include <boost/math/special_functions/gamma.hpp>
#include <armadillo>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {

  template<typename T>
  class GammaStatic : public Stochastic<T> {
    double alpha_, beta_;
  public:
    GammaStatic(const T& x, const double alpha, const double beta, const bool observed = false): Stochastic<T>(x,observed), alpha_(alpha), beta_(beta) {}
    double logp() const {
      return (Stochastic<T>::value < 0 ) ? -std::numeric_limits<double>::infinity() : accu( (alpha_ - 1.0) * log(Stochastic<T>::value) - beta_*Stochastic<T>::value - boost::math::lgamma(alpha_) + alpha_*log(beta_) );
    }
    void jump(RngBase& rng) {
      const T oldvalue(Stochastic<T>::value);
      if(Stochastic<T>::observed_) {
        return;
      } else {
        do {
          Stochastic<T>::value = oldvalue + rng.normal() * Stochastic<T>::scale_;
        } while(Stochastic<T>::value < 0);
      }
    }
  };

} // namespace cppbugs
#endif // MCMC_GAMMA_HPP

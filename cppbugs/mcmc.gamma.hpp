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
#include <armadillo>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {


  template<typename T>
  class Gamma : public Stochastic<T> {
  public:
    Gamma(const T& value, const bool observed=false): Stochastic<T>(value,observed) {}

    template<typename U, typename V>
    void dgamma(const U& alpha, const V& beta) {
      Stochastic<double>::logp_ = (Stochastic<double>::value < 0 ) ? -std::numeric_limits<double>::infinity() : accu( (alpha - 1.0) * log(Stochastic<double>::value) - beta*Stochastic<double>::value - boost::math::lgamma(alpha) + alpha*log(beta) );
    }
  };

  template<>
  class Gamma <double> : public Stochastic<double> {
  public:
    Gamma(const double& value, const bool observed=false): Stochastic<double>(value,observed) {}

    void dgamma(const double alpha, const double beta) {
      Stochastic<double>::logp_ = (Stochastic<double>::value < 0 ) ? -std::numeric_limits<double>::infinity() : (alpha - 1.0) * log(Stochastic<double>::value) - beta*Stochastic<double>::value - boost::math::lgamma(alpha) + alpha*log(beta);
    }
  };
} // namespace cppbugs
#endif // MCMC_GAMMA_HPP

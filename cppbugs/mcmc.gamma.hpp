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

    // modified jumper to only take positive jumps
    void jump(RngBase& rng) {
      for(size_t i = 0; i < Stochastic<T>::value.n_elem; i++) {
        double new_value = Stochastic<T>::value[i] + rng.normal() * Stochastic<T>::scale_;
        Stochastic<T>::value[i] = new_value > 0 ? new_value : Stochastic<T>::value[i];
      }
    }

    template<typename U, typename V>
    void dgamma(const U& alpha, const V& beta) {
      arma::umat neg_elements = find(Stochastic<T>::value < 0,1);
      Stochastic<T>::logp_ = neg_elements.n_elem ? -std::numeric_limits<double>::infinity() : accu( (alpha - 1.0) * log(Stochastic<T>::value) - beta*Stochastic<T>::value - log_gamma(alpha) + alpha*log(beta) );
    }
  };

  template<>
  class Gamma <double> : public Stochastic<double> {
  public:
    Gamma(const double& value, const bool observed=false): Stochastic<double>(value,observed) {}

    // modified jumper to only take positive jumps
    void jump(RngBase& rng) {
      double new_value = Stochastic<double>::value + rng.normal() * Stochastic<double>::scale_;
      Stochastic<double>::value = new_value > 0 ? new_value : Stochastic<double>::value;
    }

    void dgamma(const double alpha, const double beta) {
      Stochastic<double>::logp_ = (Stochastic<double>::value < 0 ) ? -std::numeric_limits<double>::infinity() : (alpha - 1.0) * log(Stochastic<double>::value) - beta*Stochastic<double>::value - log_gamma(alpha) + alpha*log(beta);
    }
  };
} // namespace cppbugs
#endif // MCMC_GAMMA_HPP

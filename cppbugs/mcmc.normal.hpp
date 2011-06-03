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

#ifndef MCMC_NORMAL_HPP
#define MCMC_NORMAL_HPP


#include <cmath>
#include <armadillo>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {

  template<typename T>
  class Normal : public Stochastic<T> {
  public:
    Normal(const T& value, const bool observed=false): Stochastic<T>(value,observed) {}

    template<typename U, typename V>
    void dnorm(const U& mu, const V& tau) {
      Stochastic<T>::logp_ = accu(0.5*log(0.5*tau/arma::math::pi()) - 0.5 * tau * pow(Stochastic<T>::value - mu,2.0));
    }

    // need this specialization b/c we need to do schur product btwn two mat's
    void dnorm(const arma::mat& mu, const arma::mat& tau) {
      Stochastic<T>::logp_ = accu(0.5*log(0.5*tau/arma::math::pi()) - 0.5 * tau % pow(Stochastic<T>::value - mu,2.0));
    }
  };

} // namespace cppbugs
#endif // MCMC_NORMAL_HPP

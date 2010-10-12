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

#include <armadillo>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {

  template<typename T>
  class Normal : public Stochastic<T> {
  public:
    Normal(const T& x, const bool observed = false): Stochastic<T>(x,observed) {}

    template<typename U, typename V>
    double logp(const U& mu, const V& tau) const {
      return accu(0.5*log(0.5*tau/arma::math::pi()) - 0.5 * tau * pow(Stochastic<T>::value - mu,2.0));
    }

    // need this specialization b/c we need to do schur product btwn two mat's
    double logp(const arma::mat& mu, const arma::mat& tau) const {
      return accu(0.5*log(0.5*tau/arma::math::pi()) - 0.5 * tau % pow(Stochastic<T>::value - mu,2.0));
    }
  };

  template<typename T>
  class NormalStatic : public Stochastic<T> {
    double mu_, tau_;
  public:
    NormalStatic(const T& x, const double mu, const double tau, const bool observed = false): Stochastic<T>(x,observed), mu_(mu), tau_(tau) {}
    double logp() const {
      return accu(0.5*log(0.5*tau_/arma::math::pi()) - 0.5 * tau_ * pow(Stochastic<T>::value - mu_,2.0));
    }
  };

} // namespace cppbugs
#endif //MCMC_NORMAL_HPP

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

#ifndef MCMC_NORMAL_HPP
#define MCMC_NORMAL_HPP


#include <cmath>
#include <armadillo>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {

  template<typename T, typename U, typename V>
  class NormalLikelihood : public LikelihoodFunctor {
  protected:
    const T& x_;
    const U& mu_;
    const V& tau_;
  public:
    NormalLikelihood(const T& x, const U& mu, const V& tau): x_(x), mu_(mu), tau_(tau) {}
    double getLikelihood() const {
      accu(0.5*log(0.5*tau_/arma::math::pi()) - 0.5 * tau_ * pow(x_ - mu_,2.0));
    }
  };

  template<typename T>
  class Normal : public Stochastic<T> {
  public:
    Normal(const T& value, const bool observed=false): Stochastic<T>(value,observed) {}

    // need this specialization b/c we need to do schur product btwn two mat's
    template<typename U, typename V>
    void dnorm(const MCMCSpecialized<U>& mu, const MCMCSpecialized<V>& tau) {
      Stochastic<T>::likelihood_functor_p = new NormalLikelihood<T,U,V>(Stochastic<T>::value, mu.value, tau.value);
    }

    void dnorm(const double& mu, const double& tau) {
      Stochastic<T>::likelihood_functor_p = new NormalLikelihood<T,double,double>(Stochastic<T>::value, mu, tau);
    }
  };

} // namespace cppbugs
#endif // MCMC_NORMAL_HPP

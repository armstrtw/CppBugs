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

#ifndef CPPBUGS_HPP
#define CPPBUGS_HPP

#include <cmath>
#include <iostream>
#include <vector>
#include <map>
#include <armadillo>
#include <boost/random.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <cppbugs/mcmc.rng.hpp>
#include <cppbugs/mcmc.object.hpp>
#include <cppbugs/mcmc.deterministic.hpp>
#include <cppbugs/mcmc.stochastic.hpp>
#include <cppbugs/mcmc.model.base.hpp>

namespace cppbugs {

  template<typename T>
  class Normal : public Stochastic<T> {
  public:
    Normal(const T& x, const bool observed = false): Stochastic<T>(x,observed) {}

    template<typename U, typename V>
    double logp(const U& mu, const V& tau) const {
      return accu(0.5*log(0.5*tau/arma::math::pi()) - 0.5 * tau * pow(Stochastic<T>::value - mu,2));
    }
  };

  template<typename T>
  class NormalStatic : public Stochastic<T> {
    double mu_, tau_;
  public:
    NormalStatic(const T& x, const double mu, const double tau, const bool observed = false): Stochastic<T>(x,observed), mu_(mu), tau_(tau) {}
    double logp() const {
      return accu(0.5*log(0.5*tau_/arma::math::pi()) - 0.5 * tau_ * pow(Stochastic<T>::value - mu_,2));
    }
  };

  template<typename T>
  class Uniform : public Stochastic<T> {
  public:
    Uniform(const T& x, const bool observed = false): Stochastic<T>(x,observed) {}
    double logp(const double lower, const double upper) const {
      return (Stochastic<T>::value < lower || Stochastic<T>::value > upper) ? -std::numeric_limits<double>::infinity() : -log(upper - lower);
    }
  };

  template<typename T>
  class UniformStatic : public Stochastic<T> {
    double lower_, upper_;
  public:
    UniformStatic(const T& x, const double lower, const double upper, const bool observed = false): Stochastic<T>(x,observed),lower_(lower),upper_(upper) {}
    double logp() const {
      return (Stochastic<T>::value < lower_ || Stochastic<T>::value > upper_) ? -std::numeric_limits<double>::infinity() : -log(upper_ - lower_);
    }
    void jump(RngBase& rng) {
      const T oldvalue(Stochastic<T>::value);
      if(Stochastic<T>::observed_) {
        return;
      } else {
        do {
          Stochastic<T>::value = oldvalue + rng.normal() * Stochastic<T>::scale_;
        } while(Stochastic<T>::value < lower_ || Stochastic<T>::value > upper_);
      }
    }
  };

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
#endif // CPPBUGS_HPP

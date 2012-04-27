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

#ifndef MCMC_BERNOULLI_HPP
#define MCMC_BERNOULLI_HPP


#include <cmath>
#include <armadillo>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {

  template <typename T,typename U>
  class BernoulliLikelihiood : public Likelihiood {
    const T& x_;
    const U& p_;
  public:
    BernoulliLikelihiood(const T& x, const U& p): x_(x), p_(p) {}
    inline double calc() const {
      return bernoulli_logp(x_,p_);
    }
  };

  template<typename T>
  class Bernoulli : public DynamicStochastic<T> {

    template<typename U>
    void bernoulli_jump(RngBase& rng, U& value, const double scale) {
      double jump_probability = 1.0 - pow(0.5,scale);
      arma::uvec flips = arma::find(arma::randu<arma::vec>(value.n_elem) < jump_probability);
      for(unsigned int i = 0; i < flips.n_elem; i++) {
        value[ flips[i] ] = value[ flips[i] ] ? 0 : 1;
      }
    }

    void bernoulli_jump(RngBase& rng, int& value, const double scale) {
      double jump_probability = 1.0 - pow(0.5,scale);
      if(rng.uniform() < jump_probability) {
        value = value ? 0 : 1;
      }
    }

    void bernoulli_jump(RngBase& rng, double& value, const double scale) {
      double jump_probability = 1.0 - pow(0.5,scale);
      if(rng.uniform() < jump_probability) {
        value = value ? 0 : 1;
      }
    }

  public:
    Bernoulli(T& value): DynamicStochastic<T>(value) {}

    void jump(RngBase& rng) {
      bernoulli_jump(rng, DynamicStochastic<T>::value, DynamicStochastic<T>::scale_);
    }

    template<typename U>
    Bernoulli<T>& dbern(const U& p) {
      Stochastic::likelihood_functor = new BernoulliLikelihiood<T,U>(DynamicStochastic<T>::value,p);
      return *this;
    }
  };

  template<typename T>
  class ObservedBernoulli : public Observed<T> {
  public:
    ObservedBernoulli(const T& value): Observed<T>(value) {}

    template<typename U>
    ObservedBernoulli<T>& dbern(const U& p) {
      Stochastic::likelihood_functor = new BernoulliLikelihiood<T,U>(Observed<T>::value,p);
      return *this;
    }
  };
} // namespace cppbugs
#endif // MCMC_BERNOULLI_HPP

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
  class Bernoulli : public DynamicStochastic<T> {
  private:
    const U& p_;
    const bool destory_p_;
    template<typename V>
    void bernoulli_jump(RngBase& rng, V& value, const double scale) {
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
    Bernoulli(T& value, const U& p): DynamicStochastic<T>(value), p_(p), destory_p_(false) { dimension_check(value, p_); }
    // special ifdef for const bug/feature introduced in gcc 4.7
#if GCC_VERSION > 40700
    Bernoulli(T& value, const U&& p): DynamicStochastic<T>(value), p_(p), destory_p_(true) { dimension_check(value, p_); }
#endif

    ~Bernoulli() {
      if(destory_p_) { delete &p_; }
    }

    void jump(RngBase& rng) {
      bernoulli_jump(rng, DynamicStochastic<T>::value, DynamicStochastic<T>::scale_);
    }
    const double loglik() const { return bernoulli_logp(DynamicStochastic<T>::value, p_); }
  };

  template <typename T,typename U>
  class ObservedBernoulli : public Observed<T> {
  private:
    const U& p_;
    const bool destory_p_;
  public:
    ObservedBernoulli(const T& value, const U& p): Observed<T>(value), p_(p), destory_p_(false) { dimension_check(Observed<T>::value, p_); }
    // special ifdef for const bug/feature introduced in gcc 4.7
#if GCC_VERSION > 40700
    ObservedBernoulli(T& value, const U&& p): Observed<T>(value), p_(p), destory_p_(true) { dimension_check(value, p_); }
#endif

    ~ObservedBernoulli() {
      if(destory_p_) { delete &p_; }
    }
    const double loglik() const { return bernoulli_logp(Observed<T>::value, p_); }
  };
} // namespace cppbugs
#endif // MCMC_BERNOULLI_HPP

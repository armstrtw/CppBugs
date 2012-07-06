///////////////////////////////////////////////////////////////////////////
// Copyright (C) 2012 Jacques-Henri Jourdan                              //
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

#ifndef MCMC_EXPONENTIAL_HPP
#define MCMC_EXPONENTIAL_HPP


#include <cmath>
#include <armadillo>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {

  template <typename T,typename U>
  class ExponentialLikelihiood : public Likelihiood {
    const T& x_;
    const U& lambda_;
  public:
    ExponentialLikelihiood(const T& x, const U& lambda): x_(x), lambda_(lambda) { dimension_check(x_, lambda_); }
    inline double calc() const {
      return arma::accu(log_approx(lambda_) - arma::schur(lambda_, x_));
    }
  };

  template<typename T>
  class Exponential : public DynamicStochastic<T> {
  public:
    Exponential(T& value): DynamicStochastic<T>(value) {}

    // modified jumper to only take positive jumps
    void jump(RngBase& rng) { positive_jump_impl(rng, DynamicStochastic<T>::value,DynamicStochastic<T>::scale_); }

    template<typename U>
    Exponential<T>& dexp(const U& lambda) {
      Stochastic::likelihood_functor = new ExponentialLikelihiood<T,U>(DynamicStochastic<T>::value,lambda);
      return *this;
    }
  };

  template<typename T>
  class ObservedExponential : public Observed<T> {
  public:
    ObservedExponential(const T& value): Observed<T>(value) {}

    template<typename U>
    ObservedExponential<T>& dexp(const U& lambda) {
      Stochastic::likelihood_functor = new ExponentialLikelihiood<T,U>(Observed<T>::value,lambda);
      return *this;
    }
  };

} // namespace cppbugs
#endif // MCMC_EXPONENTIAL_HPP

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

#ifndef MCMC_MULTIVARIATE_NORMAL_HPP
#define MCMC_MULTIVARIATE_NORMAL_HPP

#include <armadillo>
#include <cppbugs/mcmc.dynamic.stochastic.hpp>
#include <cppbugs/mcmc.observed.hpp>

namespace cppbugs {

  template <typename T,typename U>
  class MultivariateNormalLikelihiood : public Likelihiood {
    const T& x_;
    const U& mu_;
    const arma::mat& sigma_;
  public:
    MultivariateNormalLikelihiood(const T& x,  const U& mu,  const arma::mat& sigma): x_(x), mu_(mu), sigma_(sigma)
    {
      // need a modified dimension check
      dimension_check(x_, mu_);
      if(x_.n_elem != sigma_.n_rows || x_.n_elem != sigma_.n_cols) {
        throw std::logic_error("ERROR: dimensions of x do not match sigma");
      }
    }
    inline double calc() const {
      return multivariate_normal_sigma_logp(x_,mu_,sigma_);
    }
  };

  template<typename T>
  class MultivariateNormal : public DynamicStochastic<T> {
  public:
    MultivariateNormal(T& value): DynamicStochastic<T>(value) {}

    template<typename U>
    MultivariateNormal<T>& dmvnorm(const U& mu, const arma::mat& sigma) {
      Stochastic::likelihood_functor = new MultivariateNormalLikelihiood<T,U>(DynamicStochastic<T>::value,mu,sigma);
      return *this;
    }
  };

  template<typename T>
  class ObservedMultivariateNormal : public Observed<T> {
  public:
    ObservedMultivariateNormal(const T& value): Observed<T>(value) {}

    template<typename U>
    ObservedMultivariateNormal<T>& dmvnorm(const U& mu, const arma::mat& sigma) {
      Stochastic::likelihood_functor = new MultivariateNormalLikelihiood<T,U>(Observed<T>::value,mu,sigma);
      return *this;
    }
  };

} // namespace cppbugs
#endif // MCMC_MULTIVARIATE_NORMAL_HPP

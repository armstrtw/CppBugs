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

  template<typename T, typename U, typename V>
  class MultivariateNormal : public DynamicStochastic<T> {
  private:
    const U& mu_;
    const V& sigma_;
    const bool destory_mu_, destory_sigma_;
  public:
    MultivariateNormal(T& value, const U& mu, const V& sigma): DynamicStochastic<T>(value), mu_(mu), sigma_(sigma), destory_mu_(false), destory_sigma_(false) { dimension_check(value, mu_, sigma_); }
    ~MultivariateNormal() {
      if(destory_mu_) { delete &mu_; }
      if(destory_sigma_) { delete &sigma_; }
    }
    const double loglik() const { return multivariate_normal_sigma_logp(DynamicStochastic<T>::value,mu_,sigma_); }
  };

  template<typename T, typename U, typename V>
  class ObservedMultivariateNormal : public Observed<T> {
  private:
    const U& mu_;
    const V& sigma_;
    const bool destory_mu_, destory_sigma_;
  public:
    ObservedMultivariateNormal(const T& value, const U& mu, const V& sigma): Observed<T>(value), mu_(mu), sigma_(sigma), destory_mu_(false), destory_sigma_(false) { dimension_check(value, mu_, sigma_); }
    ~ObservedMultivariateNormal() {
      if(destory_mu_) { delete &mu_; }
      if(destory_sigma_) { delete &sigma_; }
    }
    const double loglik() const { return multivariate_normal_sigma_logp(Observed<T>::value,mu_,sigma_); }
  };

  template<typename T, typename U, typename V>
  class ObservedMultivariateNormalChol : public Observed<T> {
  private:
    const U& mu_;
    const V& R_;
  public:
    ObservedMultivariateNormalChol(const T& value, const U& mu, const V& R): Observed<T>(value), mu_(mu), R_(R) {}
    const double loglik() const { return multivariate_normal_chol_logp(Observed<T>::value,mu_,R_); }
  };


} // namespace cppbugs
#endif // MCMC_MULTIVARIATE_NORMAL_HPP

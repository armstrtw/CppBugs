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

#ifndef MCMC_STOCHASTIC_HPP
#define MCMC_STOCHASTIC_HPP


#include <cmath>
#include <armadillo>
#include <boost/math/special_functions/gamma.hpp>
#include <cppbugs/mcmc.specialized.hpp>
#include <cppbugs/mcmc.jump.hpp>
#include <cppbugs/mcmc.logp.hpp>

namespace cppbugs {

  double tune_factor(const double acceptance_ratio) {
    const double univariate_target_ar = 0.6;
    const double thresh = 0.1;
    const double dilution = 1.0;
    double diff = acceptance_ratio - univariate_target_ar;
    return 1.0 + diff * dilution * static_cast<double>(fabs(diff) > thresh);
  }

  template<typename T>
  class Stochastic : public MCMCSpecialized<T> {
  protected:
    bool observed_;
    double logp_,accepted_,rejected_,scale_;
  public:
    Stochastic(const T& value, const bool observed=false): MCMCSpecialized<T>(value), observed_(observed),
                                                           logp_(-std::numeric_limits<double>::infinity()),accepted_(0), rejected_(0),
                                                           scale_(1) {}
    const double* getLogp() const { return &logp_; }
    bool isDeterministc() const { return false; }
    bool isStochastic() const { return true; }
    bool isObserved() const { return observed_; }
    void jump(RngBase& rng) { jump_impl(rng,MCMCSpecialized<T>::value,scale_); }
    void accept() { accepted_ += 1; }
    void reject() { rejected_ += 1; }
    void tune() {
      double ar_ratio = accepted_ / (accepted_ + rejected_);
      scale_ *= tune_factor(ar_ratio);
      accepted_ = 0;
      rejected_ = 0;
    }

    template<typename U, typename V>
    void dnorm(const U& mu, const V& tau) {
      logp_ = accu(0.5*log(0.5*tau/arma::math::pi()) - 0.5 * tau * pow(MCMCSpecialized<T>::value - mu,2.0));
    }

    // need this specialization b/c we need to do schur product btwn two mat's
    void dnorm(const arma::mat& mu, const arma::mat& tau) {
      logp_ = accu(0.5*log(0.5*tau/arma::math::pi()) - 0.5 * tau % pow(MCMCSpecialized<T>::value - mu,2.0));
    }

    void dunif(const double lower, const double upper) {
      logp_ = dunif_impl(MCMCSpecialized<T>::value,lower,upper);
    }

    template<typename U, typename V>
    void dgamma(const U& alpha, const V& beta) {
      logp_ = (MCMCSpecialized<T>::value < 0 ) ? -std::numeric_limits<double>::infinity() : accu( (alpha - 1.0) * log(MCMCSpecialized<T>::value) - beta*MCMCSpecialized<T>::value - boost::math::lgamma(alpha) + alpha*log(beta) );
    }

    template<typename U, typename V>
    void dbinom(const U& n, const V& p) {
      arma::uvec less_than_zero = find(MCMCSpecialized<T>::value < 0,1);
      arma::uvec greater_than_n = find(MCMCSpecialized<T>::value > n,1);

      if(less_than_zero.n_elem) {
        logp_ = -std::numeric_limits<double>::infinity();
      }

      if(greater_than_n.n_elem) {
        logp_ = -std::numeric_limits<double>::infinity();
      }

      logp_ = accu(MCMCSpecialized<T>::value % log(p) + (n-MCMCSpecialized<T>::value) % log(1-p) + factln(n)-factln(MCMCSpecialized<T>::value)-factln(n-MCMCSpecialized<T>::value));
    }
  };

} // namespace cppbugs
#endif // MCMC_STOCHASTIC_HPP

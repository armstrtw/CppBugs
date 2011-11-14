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

#ifndef MCMC_STOCHASTIC_HPP
#define MCMC_STOCHASTIC_HPP


#include <cmath>
#include <armadillo>
#include <cppbugs/mcmc.specialized.hpp>
#include <cppbugs/mcmc.jump.hpp>
#include <cppbugs/mcmc.likelihood.functor.hpp>

namespace cppbugs {

  template<typename T>
  class Stochastic : public MCMCSpecialized<T> {
  protected:
    bool observed_;
    double logp_,accepted_,rejected_,scale_;
    LikelihoodFunctor* likelihood_functor_p;
  public:
    Stochastic(const T& value, const bool observed=false):
      MCMCSpecialized<T>(value), observed_(observed),
      logp_(-std::numeric_limits<double>::infinity()),accepted_(0), rejected_(0),
      scale_(1), likelihood_functor_p(NULL)
    {
      // don't need to save history of observed variables
      if(observed_) {
        MCMCSpecialized<T>::setSaveHistory(false);
      }
    }
    virtual ~Stochastic() { delete likelihood_functor_p; }
    const double* getLogp() const { return &logp_; }
    bool isDeterministc() const { return false; }
    bool isStochastic() const { return true; }
    bool isObserved() const { return observed_; }
    void jump(RngBase& rng) { jump_impl(rng,MCMCSpecialized<T>::value,scale_); }
    void accept() { accepted_ += 1; }
    void reject() { rejected_ += 1; }
    double tune_factor(const double acceptance_ratio) {
      const double univariate_target_ar = 0.6;
      const double thresh = 0.1;
      const double dilution = 1.0;
      double diff = acceptance_ratio - univariate_target_ar;
      return 1.0 + diff * dilution * static_cast<double>(fabs(diff) > thresh);
    }
    void tune() {
      double ar_ratio = accepted_ / (accepted_ + rejected_);
      scale_ *= tune_factor(ar_ratio);
      accepted_ = 0;
      rejected_ = 0;
    }
    const double logp() const {
      return logp_;
    }
    void setScale(const double scale) {
      scale_ = scale;
    }
    double loglik() const {
      return Stochastic<T>::likelihood_functor_p ? Stochastic<T>::likelihood_functor_p->getLikelihood() : 0;
    }
  };

} // namespace cppbugs
#endif // MCMC_STOCHASTIC_HPP

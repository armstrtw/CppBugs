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

#include <cppbugs/mcmc.specialized.hpp>

namespace cppbugs {
  double accu(const double x) {
    return x;
  }

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
    double accepted_,rejected_,scale_;
  public:
    Stochastic(const T& value, const bool observed): MCMCSpecialized<T>(value), observed_(observed),
						     accepted_(0), rejected_(0),
						     scale_(1) {}
    bool isDeterministc() const { return false; }
    bool isStochastic() const { return true; }
    bool isObserved() const { return observed_; }
    void jump(RngBase& rng) {
      if(observed_) {
        return;
      } else {
        for(size_t i = 0; i < MCMCSpecialized<T>::value.n_elem; i++) {
          MCMCSpecialized<T>::value[i] += rng.normal() * scale_;
        }
      }
    }
    void accept() { accepted_ += 1; }
    void reject() { rejected_ += 1; }
    void tune() {
      double ar_ratio = accepted_ / (accepted_ + rejected_);
      scale_ *= tune_factor(ar_ratio);
      accepted_ = 0;
      rejected_ = 0;
    }
  };

  template<>
  class Stochastic<double> : public MCMCSpecialized<double> {
  protected:
    bool observed_;
    double accepted_,rejected_,scale_;
  public:
    Stochastic(const double& value, const bool observed): MCMCSpecialized<double>(value), observed_(observed),
						     accepted_(0), rejected_(0),scale_(1) {}
    bool isDeterministc() const { return false; }
    bool isStochastic() const { return true; }
    bool isObserved() const { return observed_; }
    void jump(RngBase& rng) {
      MCMCSpecialized<double>::value += rng.normal() * scale_;
    }
    void accept() { accepted_ += 1; }
    void reject() { rejected_ += 1; }
    void tune() {
      double ar_ratio = accepted_ / (accepted_ + rejected_);
      scale_ *= tune_factor(ar_ratio);
      accepted_ = 0;
      rejected_ = 0;
    }
  };
} // namespace cppbugs
#endif // MCMC_STOCHASTIC_HPP

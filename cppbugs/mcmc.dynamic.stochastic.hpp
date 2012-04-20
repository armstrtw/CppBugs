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

#ifndef MCMC_DYNAMIC_STOCHASTIC_HPP
#define MCMC_DYNAMIC_STOCHASTIC_HPP

#include <armadillo>
#include <cppbugs/mcmc.specialized.hpp>
#include <cppbugs/mcmc.stochastic.hpp>
#include <cppbugs/mcmc.jump.hpp>
#include <cppbugs/mcmc.math.hpp>

namespace cppbugs {

  template<typename T>
  class DynamicStochastic : public Dynamic<T>, public Stochastic  {
  protected:
    bool observed_;
    double accepted_,rejected_,scale_,target_ar_;

    double tune_factor(const double acceptance_ratio) {
      const double thresh = 0.1;
      const double dilution = 1.0;
      double diff = acceptance_ratio - target_ar_;
      return 1.0 + diff * dilution * static_cast<double>(fabs(diff) > thresh);
    }
  public:
    DynamicStochastic(T& value): Dynamic<T>(value), accepted_(0), rejected_(0) {
      const double scale_num = 2.38;
      double ideal_scale = sqrt(scale_num / pow(dim_size(Dynamic<T>::value),2));
      scale_ = ideal_scale > 1.0 ? 1.0 : ideal_scale;

      // heuristic to set the target acceptance ratio based on the size of the object
      // limiting the target ar to the theoretical asymptotic minimum
      target_ar_ = std::max(1/log2(dim_size(Dynamic<T>::value) + 3),0.234);
    }
    virtual ~DynamicStochastic() {}
    void jump(RngBase& rng) { jump_impl(rng,Dynamic<T>::value,scale_); }
    void accept() { accepted_ += 1; }
    void reject() { rejected_ += 1; }
    void tune() {
      double ar_ratio = accepted_ / (accepted_ + rejected_);
      scale_ *= tune_factor(ar_ratio);
      accepted_ = 0;
      rejected_ = 0;
    }
    // in Dynamic: void preserve()
    // in Dynamic: void revert()
    // in Dynamic: void tally()
    bool isDeterministc() const { return false; }
    bool isStochastic() const { return true; }
    bool isObserved() const { return false; }
    void setScale(const double scale) { scale_ = scale; }
  };

} // namespace cppbugs
#endif // MCMC_DYNAMIC_STOCHASTIC_HPP

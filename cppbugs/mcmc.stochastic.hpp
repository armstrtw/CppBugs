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

namespace cppbugs {

  template<typename T>
  class Stochastic : public MCMCSpecialized<T> {
  protected:
    bool observed_;
    double logp_,accepted_,rejected_,scale_;

    template<typename U>
    double accu(const U&  x) {
      return accu(x);
    }

    double accu(const double x) {
      return x;
    }

    double factln_single(int n) {
      if(n > 100) {
	return boost::math::lgamma(static_cast<double>(n) + 1);
      }
      double ans(1);
      for (int i=n; i>1; i--) {
	ans *= i;
      }
      return log(ans);
    }

    double factln(const int i) {
      static std::vector<double> factln_table;

      if(i < 0) {
	return -std::numeric_limits<double>::infinity();
      }

      if(factln_table.size() < static_cast<size_t>(i+1)) {
	for(int j = factln_table.size(); j < (i+1); j++) {
	  factln_table.push_back(factln_single(j));
	}
      }
      //return factln_table.at(i);
      return factln_table[i];
    }

    arma::mat factln(const arma::imat& x) {
      arma::mat ans; ans.copy_size(x);
      for(size_t i = 0; i < x.n_elem; i++) {
	ans[i] = factln(x[i]);
      }
      return ans;
    }
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
  };

} // namespace cppbugs
#endif // MCMC_STOCHASTIC_HPP

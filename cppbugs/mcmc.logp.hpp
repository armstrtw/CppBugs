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

#ifndef MCMC_LOGP_HPP
#define MCMC_LOGP_HPP

#include <cmath>
#include <armadillo>
#include <boost/math/special_functions/gamma.hpp>

namespace cppbugs {
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
    //return factln_table[i];
    return factln_table.at(i);
  }

  arma::mat factln(const arma::imat& x) {
    arma::mat ans; ans.copy_size(x);
    for(size_t i = 0; i < x.n_elem; i++) {
      ans[i] = factln(x[i]);
    }
    return ans;
  }

  double dunif_impl(const double value, const double lower, const double upper) {
    return (value < lower || value > upper) ? -std::numeric_limits<double>::infinity() : -log(upper - lower);
  }

  double dunif_impl(const arma::mat& value, const double lower, const double upper) {
    arma::uvec less_than_lower_bound = find(value < lower,1);
    arma::uvec greater_than_upper_bound = find(value > upper,1);

    if(less_than_lower_bound.n_elem || greater_than_upper_bound.n_elem) {
      return -std::numeric_limits<double>::infinity();
    }

    return accu(-log(upper - lower));
  }
} // namespace cppbugs
#endif // MCMC_LOGP_HPP

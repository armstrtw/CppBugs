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

#ifndef MCMC_BINOMIAL_HPP
#define MCMC_BINOMIAL_HPP

#include <vector>
#include <armadillo>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {

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

  template<typename T>
  class Binomial : public Stochastic<T> {
  public:
    Binomial(const T& x, const bool observed = false): Stochastic<T>(x,observed) {}

    template<typename U, typename V>
    void logp(const U& n, const V& p) {
      arma::uvec less_than_zero = find(Stochastic<T>::value < 0,1);
      arma::uvec greater_than_n = find(Stochastic<T>::value > n,1);

      if(less_than_zero.n_elem) {
        Stochastic<T>::logp_ = -std::numeric_limits<double>::infinity();
      }
      
      if(greater_than_n.n_elem) {
        Stochastic<T>::logp_ = -std::numeric_limits<double>::infinity();
      }

      Stochastic<T>::logp_ = accu(Stochastic<T>::value % log(p) + (n-Stochastic<T>::value) % log(1-p) + factln(n)-factln(Stochastic<T>::value)-factln(n-Stochastic<T>::value));
    }
  };

} // namespace cppbugs
#endif //MCMC_BINOMIAL_HPP

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

#ifndef MCMC_LIKELIHOOD_FUNCTOR_HPP
#define MCMC_LIKELIHOOD_FUNCTOR_HPP


#include <cmath>
#include <armadillo>
#include <boost/math/special_functions/gamma.hpp>
#include <cppbugs/mcmc.specialized.hpp>
#include <cppbugs/mcmc.jump.hpp>

namespace cppbugs {

  class LikelihoodFunctor {
  protected:
    template<typename U>
    double accu(const U&  x) {
      return arma::accu(x);
    }

    static double accu(const double x) {
      return x;
    }

    double log_gamma(const double x) {
      return boost::math::lgamma(x);
    }

    double factln_single(int n) {
      if(n > 100) {
	return log_gamma(static_cast<double>(n) + 1);
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
    LikelihoodFunctor() {}
    virtual double getLikelihood() const = 0;
  };

  template<typename T, typename U, typename V>
  class NormalLikelihood : public LikelihoodFunctor {
  protected:
    const T& x_;
    const U& mu_;
    const V& tau_;
  public:
    NormalLikelihood(const T& x, const U& mu, const V& tau): x_(x), mu_(mu), tau_(tau) {}
    double getLikelihood() const {
      accu(0.5*log(0.5*tau_/arma::math::pi()) - 0.5 * tau_ * pow(x_ - mu_,2.0));
    }
  };

} // namespace cppbugs
#endif // MCMC_LIKELIHOOD_FUNCTOR_HPP

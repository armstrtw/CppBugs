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

#ifndef MCMC_UNIFORM_HPP
#define MCMC_UNIFORM_HPP


#include <cmath>
#include <armadillo>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {

  template<typename T>
  class Uniform : public Stochastic<T> {
  public:
    Uniform(const T& value, const bool observed=false): Stochastic<T>(value,observed) {}
    void dunif(const double lower, const double upper) {
      Stochastic<T>::logp_ =  (Stochastic<T>::value < lower || Stochastic<T>::value > upper) ? -std::numeric_limits<double>::infinity() : -log(upper - lower);
    }
  };

  template<>
  class Uniform <double> : public Stochastic<double> {
  public:
    Uniform(const double& value, const bool observed=false): Stochastic<double>(value,observed) {}
    void dunif(const double lower, const double upper) {
      Stochastic<double>::logp_ =  (Stochastic<double>::value < lower || Stochastic<double>::value > upper) ? -std::numeric_limits<double>::infinity() : -log(upper - lower);
    }
  };


  template<>
  class Uniform <arma::mat> : public Stochastic<arma::mat> {
  public:
    Uniform(const arma::mat& value, const bool observed=false): Stochastic<arma::mat>(value,observed) {}

    template<typename U, typename V>
    void dunif(const U& lower, const V& upper) {
      arma::uvec less_than_lower_bound = find(value < lower,1);
      arma::uvec greater_than_upper_bound = find(value > upper,1);

      if(less_than_lower_bound.n_elem || greater_than_upper_bound.n_elem) {
	Stochastic<arma::mat>::logp_ = -std::numeric_limits<double>::infinity();
      } else {
	Stochastic<arma::mat>::logp_ = accu(-log(upper - lower));
      }
    }
  };
} // namespace cppbugs
#endif // MCMC_UNIFORM_HPP

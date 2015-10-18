///////////////////////////////////////////////////////////////////////////
// Copyright (C) 2011 Jacques-Henri Jourdan                              //
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

#pragma once


#include <cmath>
#include <armadillo>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {

  class DiscreteLikelihiood : public Likelihiood {
    const int& x_;
    const arma::vec& p_;
  public:
    DiscreteLikelihiood(const int& x, const arma::vec& p): x_(x), p_(p) { }
    inline double calc() const {
      if(x_ < 0 || x_ >= (int)p_.n_elem)
	return -std::numeric_limits<double>::infinity();
      return log_approx(p_[x_]);
    }
  };

  template<typename T> class Discrete;

  template<>
  class Discrete<int> : public DynamicStochastic<int> {
  public:
    Discrete(int& value): DynamicStochastic<int>(value) {}

    Discrete<int>& ddiscr(const arma::vec& p) {
      Stochastic::likelihood_functor = new DiscreteLikelihiood(DynamicStochastic<int>::value,p);
      return *this;
    }
  };

  template<typename T> class ObservedDiscrete;

  template<>
  class ObservedDiscrete<int> : public Observed<int> {
  public:
    ObservedDiscrete(const int& value): Observed<int>(value) {}

    ObservedDiscrete<int>& ddiscr(const arma::vec& p) {
      Stochastic::likelihood_functor = new DiscreteLikelihiood(Observed<int>::value,p);
      return *this;
    }
  };
} // namespace cppbugs

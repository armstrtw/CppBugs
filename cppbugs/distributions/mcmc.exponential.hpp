///////////////////////////////////////////////////////////////////////////
// Copyright (C) 2012 Jacques-Henri Jourdan                              //
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

#ifndef MCMC_EXPONENTIAL_HPP
#define MCMC_EXPONENTIAL_HPP


#include <cmath>
#include <armadillo>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {

  template<typename T, typename U>
  class Exponential : public DynamicStochastic<T> {
  private:
    const U& lambda_;
  public:
    Exponential(T& value, const U& lambda): DynamicStochastic<T>(value), lambda_(lambda) { dimension_check(value,lambda); }

    // modified jumper to only take positive jumps
    void jump(RngBase& rng) { positive_jump_impl(rng, DynamicStochastic<T>::value,DynamicStochastic<T>::scale_); }
    const double loglik() const { return exponential_logp(DynamicStochastic<T>::value,lambda_); }
  };

  template<typename T, typename U>
  class ObservedExponential : public Observed<T> {
  private:
    const U& lambda_;
  public:
    ObservedExponential(const T& value, const U& lambda): Observed<T>(value), lambda_(lambda) { dimension_check(value,lambda); }
    const double loglik() const { return exponential_logp(DynamicStochastic<T>::value,lambda_); }
  };

} // namespace cppbugs
#endif // MCMC_EXPONENTIAL_HPP

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

#ifndef MCMC_GAMMA_HPP
#define MCMC_GAMMA_HPP


#include <cmath>
#include <armadillo>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {

  template<typename T>
  class Gamma : public DynamicStochastic<T> {
  public:
    Gamma(T& value): DynamicStochastic<T>(value) {}

    // modified jumper to only take positive jumps
    void jump(RngBase& rng) { positive_jump_impl(rng, DynamicStochastic<T>::value,DynamicStochastic<T>::scale_); }

    template<typename U, typename V>
    Gamma<double>& dgamma(const U& alpha, const V& beta) {
      const T& x = DynamicStochastic<T>::value;
      Stochastic::likelihood_functor = [&x,&alpha,&beta]() {
        return gamma_logp(x,alpha,beta);
      };
      return *this;
    }
  };

} // namespace cppbugs
#endif // MCMC_GAMMA_HPP

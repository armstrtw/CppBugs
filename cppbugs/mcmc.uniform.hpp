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

#ifndef MCMC_UNIFORM_HPP
#define MCMC_UNIFORM_HPP


#include <cmath>
#include <armadillo>
#include <cppbugs/mcmc.stochastic.hpp>

namespace cppbugs {

  template <typename T,typename U, typename V>
  class UniformLikelihiood : public Likelihiood {
    const T& x_;
    const U& lower_;
    const V& upper_;
  public:
    UniformLikelihiood(const T& x,  const U& lower,  const V& upper): x_(x), lower_(lower), upper_(upper) { dimension_check(x_, lower_, upper_); }
    inline double calc() const {
      return uniform_logp(x_,lower_,upper_);
    }
  };

  template<typename T>
  class Uniform : public DynamicStochastic<T> {
  public:
    Uniform(T& value): DynamicStochastic<T>(value) {}

    template<typename U, typename V>
    Uniform<T>& dunif(const U& lower, const V& upper) {
      Stochastic::likelihood_functor = new UniformLikelihiood<T,U,V>(DynamicStochastic<T>::value,lower,upper);
      return *this;
    }
  };

  template<typename T>
  class ObservedUniform : public Observed<T> {
  public:
    ObservedUniform(const T& value): Observed<T>(value) {}

    template<typename U, typename V>
    ObservedUniform<T>& dunif(const U& lower, const V& upper) {
      Stochastic::likelihood_functor = new UniformLikelihiood<T,U,V>(Observed<T>::value,lower,upper);
      return *this;
    }
  };


} // namespace cppbugs
#endif // MCMC_UNIFORM_HPP

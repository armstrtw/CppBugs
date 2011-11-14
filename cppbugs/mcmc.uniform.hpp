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


  template<typename T, typename U, typename V>
  class UniformLikelihood : public LikelihoodFunctor {
  protected:
    const T& x;
    const U& lower;
    const V& upper;
  public:
    UniformLikelihood(const T& x_, const U& lower_, const V& upper_): x(x_), lower(lower_), upper(upper_) {}
    double getLikelihood() const {
      return (x < lower || x > upper) ? -std::numeric_limits<double>::infinity() : -accu(log(upper - lower));
    }
  };

  template<typename T>
  class Uniform : public Stochastic<T> {
  public:
    Uniform(const T& value, const bool observed=false): Stochastic<T>(value,observed) {}

    template<typename U, typename V>
    void dunif(const MCMCSpecialized<U>& lower, MCMCSpecialized<V>& upper) {
      Stochastic<T>::likelihood_functor_p = new UniformLikelihood<T,U,V>(Stochastic<T>::value, lower.value, upper.value);
    }

    void dunif(const double& lower, const double& upper) {
      Stochastic<T>::likelihood_functor_p = new UniformLikelihood<T,double,double>(Stochastic<T>::value, lower, upper);
    }
  };
} // namespace cppbugs
#endif // MCMC_UNIFORM_HPP

///////////////////////////////////////////////////////////////////////////
// Copyright (C) 2014 Whit Armstrong                                     //
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

#ifndef MCMC_DISTRIBUTIONS_HPP
#define MCMC_DISTRIBUTIONS_HPP

#include <cppbugs/mcmc.stochastic.1p.family.hpp>
#include <cppbugs/mcmc.stochastic.2p.family.hpp>

namespace cppbugs {

template <class T,class U,class V> using Normal = Stochastic2p<T,U,V,normal_logp>;
template <class T,class U,class V> using ObservedNormal = ObservedStochastic2p<T,U,V,normal_logp>;

template <class T,class U,class V> using Uniform = Stochastic2p<T,U,V,uniform_logp>;
template <class T,class U,class V> using ObservedUniform = ObservedStochastic2p<T,U,V,uniform_logp>;

// modified jumper to only take jumps on (0,1) interval
// FIXME: void jump(RngBase& rng) { bounded_jump_impl(rng, DynamicStochastic<T>::value, DynamicStochastic<T>::scale_, 0, 1); }
template <class T,class U,class V> using Beta = Stochastic2p<T,U,V,beta_logp>;
template <class T,class U,class V> using ObservedBeta = ObservedStochastic2p<T,U,V,beta_logp>;

template <class T,class U,class V> using Binomial = Stochastic2p<T,U,V,binomial_logp>;
template <class T,class U,class V> using ObservedBinomial = ObservedStochastic2p<T,U,V,binomial_logp>;

// modified jumper to only take positive jumps
// FIXME: void jump(RngBase& rng) { positive_jump_impl(rng, DynamicStochastic<T>::value, DynamicStochastic<T>::scale_); }
template <class T,class U,class V> using Gamma = Stochastic2p<T,U,V,gamma_logp>;
template <class T,class U,class V> using ObservedGamma = ObservedStochastic2p<T,U,V,gamma_logp>;

// FIXME: dimension check will not work on this
template <class T,class U,class V> using MultivariateNormal = Stochastic2p<T,U,V,multivariate_normal_sigma_logp>;
template <class T,class U,class V> using ObservedMultivariateNormal = ObservedStochastic2p<T,U,V,multivariate_normal_sigma_logp>;

// FIXME: dimension check will not work on this
template <class T,class U,class V> using MultivariateNormalChol = Stochastic2p<T,U,V,multivariate_normal_chol_logp>;
template <class T,class U,class V> using ObservedMultivariateNormalChol = ObservedStochastic2p<T,U,V,multivariate_normal_chol_logp>;


// modified jumper to only take positive jumps
// FIXME: void jump(RngBase& rng) { positive_jump_impl(rng, DynamicStochastic<T>::value,DynamicStochastic<T>::scale_); }
template <class T,class U> using Exponential = Stochastic1p<T,U,exponential_logp>;
template <class T,class U> using ObservedExponential = ObservedStochastic1p<T,U,exponential_logp>;

template <class T,class U> using Poisson = Stochastic1p<T,U,poisson_logp>;
template <class T,class U> using ObservedPoisson = ObservedStochastic1p<T,U,poisson_logp>;


template <class T,class U> using Categorical = Stochastic1p<T,U,categorical_logp>;
template <class T,class U> using ObservedCategorical = ObservedStochastic1p<T,U,categorical_logp>;


} // namespace cppbugs
#endif // MCMC_DISTRIBUTIONS_HPP

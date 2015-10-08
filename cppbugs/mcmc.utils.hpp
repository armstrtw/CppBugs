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

#ifndef MCMC_UTILS_HPP
#define MCMC_UTILS_HPP

#include <stdexcept>
#include <armadillo>

namespace cppbugs {

  double dim_size(const double x) {
    return 1;
  }

  double dim_size(const int x) {
    return 1;
  }

  double dim_size(const bool x) {
    return 1;
  }

  double dim_size(const arma::subview_elem2<double, arma::Mat<arma::uword>, arma::Mat<arma::uword> >& x) {
    arma::mat m(x);
    return m.n_elem;
  }

  double dim_size(const arma::subview_elem1<double, arma::Mat<arma::uword> >& x) {
    arma::mat m(x);
    return m.n_elem;
  }

  template<typename T>
  double dim_size(const T& x) {
    return x.n_elem;
  }

  template<typename T, typename U, typename V>
  void dimension_check(const T& x, const U& hyper1, const V& hyper2) {
    if(dim_size(hyper1) > dim_size(x) || dim_size(hyper2) > dim_size(x)) {
      throw std::logic_error("ERROR: dimensions of hyperparmeters are larger than the stochastic variable itself (is this really what you wanted to do?)");
    }
  }

  template<typename T, typename U>
  void dimension_check(const T& x, const U& hyper1) {
    if(dim_size(hyper1) > dim_size(x)) {
      throw std::logic_error("ERROR: dimensions of hyperparmeters are larger than the stochastic variable itself (is this really what you wanted to do?)");
    }
  }

} // namespace cppbugs
#endif // MCMC_UTILS_HPP

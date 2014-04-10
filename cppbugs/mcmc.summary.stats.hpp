///////////////////////////////////////////////////////////////////////////
// Copyright (C) 2012 Whit Armstrong                                     //
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

#ifndef MCMC_SUMMARY_STATS_HPP
#define MCMC_SUMMARY_STATS_HPP

#include <exception>
#include <armadillo>

namespace cppbugs {

  template<typename T>
  class initValue;

  template<>
  class initValue<double> {
  public:
    typedef double ansT;
    static const ansT init(const double x) {
      return 0;
    }
  };

  template<>
  class initValue<int> {
  public:
    typedef double ansT;
    static const ansT init(const int x) {
      return 0;
    }
  };

  template<>
  class initValue<arma::vec> {
  public:
    typedef arma::vec ansT;
    static const ansT init(const arma::vec& x) {
      return arma::zeros<arma::vec>(x.n_elem);
    }
  };

  template<>
  class initValue<arma::ivec> {
  public:
    typedef arma::vec ansT;
    static const ansT init(const arma::ivec& x) {
      return arma::zeros<arma::vec>(x.n_elem);
    }
  };

  template<>
  class initValue<arma::imat> {
  public:
    typedef arma::mat ansT;
    static const ansT init(const arma::imat& x) {
      return arma::zeros<arma::mat>(x.n_rows,x.n_cols);
    }
  };

  template<>
  class initValue<arma::mat> {
  public:
    typedef arma::mat ansT;
    static const ansT init(const arma::mat& x) {
      return arma::zeros<arma::mat>(x.n_rows,x.n_cols);
    }
  };

  template<typename T>
  typename initValue< typename std::iterator_traits<T>::value_type >::ansT mean(T beg, T end) {
    typedef typename initValue< typename std::iterator_traits<T>::value_type >::ansT ansT;
    const double len = static_cast<double>(std::distance(beg,end));
    if(len==0) { throw std::logic_error("mean: no observations."); }
    ansT ans(initValue< typename std::iterator_traits<T>::value_type >::init(*beg));

    while(beg != end) {
      ans += *beg;
      ++beg;
    }
    return ans / len;
  }

  template<typename T>
  typename initValue< typename std::iterator_traits<T>::value_type >::ansT sd(T beg, T end) {
    typedef typename initValue< typename std::iterator_traits<T>::value_type >::ansT ansT;
    const size_t n = std::distance(beg,end);
    if(n < 2) { throw std::logic_error("sd: need more than 1 observation."); }
    const double n1 = static_cast<double>(n - 1);
    ansT x_mean = mean(beg,end);
    ansT sum_squares = initValue< typename std::iterator_traits<T>::value_type >::init(*beg);

    while(beg != end) {
      sum_squares += square(*beg - x_mean);
      ++beg;
    }
    return sqrt(sum_squares / n1 );
  }

} // namespace cppbugs
#endif // MCMC_STOCHASTIC_HPP

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

#ifndef MCMC_MATH_HPP
#define MCMC_MATH_HPP

#include <stdexcept>
#include <cmath>
#include <armadillo>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/factorials.hpp>

namespace cppbugs {

  ////////////////////////////////////////////////////////////////////////////////////
  // ICSIlog Copyright taken from ICSIlog source                                    //
  // Copyright (C) 2007 International Computer Science Institute                    //
  // 1947 Center Street, Suite 600                                                  //
  // Berkeley, CA 94704                                                             //
  //                                                                                //
  // Contact information:                                                           //
  //    Oriol Vinyals	vinyals@icsi.berkeley.edu                                   //
  //    Gerald Friedland 	fractor@icsi.berkeley.edu                           //
  //                                                                                //
  // This program is free software; you can redistribute it and/or modify           //
  // it under the terms of the GNU General Public License as published by           //
  // the Free Software Foundation; either version 2 of the License, or              //
  // (at your option) any later version.                                            //
  //                                                                                //
  // This program is distributed in the hope that it will be useful,                //
  // but WITHOUT ANY WARRANTY; without even the implied warranty of                 //
  // MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                  //
  // GNU General Public License for more details.                                   //
  //                                                                                //
  // You should have received a copy of the GNU General Public License along        //
  // with this program; if not, write to the Free Software Foundation, Inc.,        //
  // 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.                    //
  //                                                                                //
  // Authors                                                                        //
  // -------                                                                        //
  //                                                                                //
  // Oriol Vinyals		vinyals@icsi.berkeley.edu                           //
  // Gerald Friedland 	fractor@icsi.berkeley.edu                                   //
  //                                                                                //
  // Acknowledgements                                                               //
  // ----------------                                                               //
  //                                                                                //
  // Thanks to Harrison Ainsworth (hxa7241@gmail.com) for his idea that             //
  // doubled the accuracy.                                                          //
  ////////////////////////////////////////////////////////////////////////////////////


  /* ICSIlog V 2.0 */
  const std::vector<float> fill_icsi_log_table2(const unsigned int precision) {
    std::vector<float> pTable(static_cast<size_t>(pow(2,precision)));

    /* step along table elements and x-axis positions
       (start with extra half increment, so the steps intersect at their midpoints.) */
    float oneToTwo = 1.0f + (1.0f / (float)( 1 <<(precision + 1) ));
    for(int i = 0;  i < (1 << precision);  ++i ) {
      // make y-axis value for table element
      pTable[i] = logf(oneToTwo) / 0.69314718055995f;
      oneToTwo += 1.0f / (float)( 1 << precision );
    }
    return pTable;
  }

  /* ICSIlog v2.0 */
  inline double icsi_log(const double vald) {
    const float val = static_cast<float>(vald);
    const unsigned int precision(10);
    static std::vector<float> pTable = fill_icsi_log_table2(precision);

    /* get access to float bits */
    register const int* const pVal = (const int*)(&val);

    /* extract exponent and mantissa (quantized) */
    register const int exp = ((*pVal >> 23) & 255) - 127;
    register const int man = (*pVal & 0x7FFFFF) >> (23 - precision);

    /* exponent plus lookup refinement */
    return static_cast<double>(((float)(exp) + pTable[man]) * 0.69314718055995f);
  }

  inline double log_approx(const double x) {
    return x <= 0 ? -std::numeric_limits<double>::infinity() : icsi_log(x);
  }
}

namespace arma {
  // log_approx
  class eop_log_approx : public eop_core<eop_log_approx> {};

  template<> template<typename eT> arma_hot arma_pure arma_inline eT
  eop_core<eop_log_approx>::process(const eT val, const eT  ) {
    return cppbugs::log_approx(val);
  }

  // Base
  template<typename T1>
  arma_inline
  const eOp<T1, eop_log_approx> log_approx(const Base<typename T1::elem_type,T1>& A) {
    arma_extra_debug_sigprint();
    return eOp<T1, eop_log_approx>(A.get_ref());
  }

  // BaseCube
  template<typename T1>
  arma_inline
  const eOpCube<T1, eop_log_approx> log_approx(const BaseCube<typename T1::elem_type,T1>& A) {
    arma_extra_debug_sigprint();
    return eOpCube<T1, eop_log_approx>(A.get_ref());
  }
}


namespace arma {
  // lgamma
  class eop_lgamma : public eop_core<eop_lgamma> {};

  template<> template<typename eT> arma_hot arma_pure arma_inline eT
  eop_core<eop_lgamma>::process(const eT val, const eT  ) {
    return boost::math::lgamma(val);
  }

  // Base
  template<typename T1>
  arma_inline
  const eOp<T1, eop_lgamma> lgamma(const Base<typename T1::elem_type,T1>& A) {
    arma_extra_debug_sigprint();
    return eOp<T1, eop_lgamma>(A.get_ref());
  }

  // BaseCube
  template<typename T1>
  arma_inline
  const eOpCube<T1, eop_lgamma> lgamma(const BaseCube<typename T1::elem_type,T1>& A) {
    arma_extra_debug_sigprint();
    return eOpCube<T1, eop_lgamma>(A.get_ref());
  }
}


namespace arma {
  // factln

  double factln(const int i) {
    static std::vector<double> factln_table;

    if(i < 0) {
      return -std::numeric_limits<double>::infinity();
    }

    if(i > 100) {
      return boost::math::lgamma(static_cast<double>(i) + 1);
    }

    if(factln_table.size() < static_cast<size_t>(i+1)) {
      for(int j = factln_table.size(); j < (i+1); j++) {
        factln_table.push_back(std::log(boost::math::factorial<double>(static_cast<double>(j))));
      }
    }
    //for(auto v : factln_table) { std::cout << v << "|"; }  std::cout << std::endl;
    return factln_table[i];
  }

  class eop_factln : public eop_core<eop_factln> {};

  template<> template<typename eT> arma_hot arma_pure arma_inline eT
  eop_core<eop_factln>::process(const eT val, const eT  ) {
    return factln(val);
  }

  // Base
  template<typename T1>
  arma_inline
  const eOp<T1, eop_factln> factln(const Base<typename T1::elem_type,T1>& A) {
    arma_extra_debug_sigprint();
    return eOp<T1, eop_factln>(A.get_ref());
  }

  // BaseCube
  template<typename T1>
  arma_inline
  const eOpCube<T1, eop_factln> factln(const BaseCube<typename T1::elem_type,T1>& A) {
    arma_extra_debug_sigprint();
    return eOpCube<T1, eop_factln>(A.get_ref());
  }
}


namespace arma {

  template<typename T1, typename T2>
  arma_inline
  const eGlue<T1, T2, eglue_schur>
  schur(const Base<typename T1::elem_type,T1>& X, const Base<typename T1::elem_type,T2>& Y) {
    arma_extra_debug_sigprint();
    return eGlue<T1, T2, eglue_schur>(X.get_ref(), Y.get_ref());
  }

  //! element-wise multiplication of Base objects with different element types
  template<typename T1, typename T2>
  inline
  const mtGlue<typename promote_type<typename T1::elem_type, typename T2::elem_type>::result, T1, T2, glue_mixed_schur>
  schur
  (
   const Base< typename force_different_type<typename T1::elem_type, typename T2::elem_type>::T1_result, T1>& X,
   const Base< typename force_different_type<typename T1::elem_type, typename T2::elem_type>::T2_result, T2>& Y
   )
  {
    arma_extra_debug_sigprint();
    typedef typename T1::elem_type eT1;
    typedef typename T2::elem_type eT2;
    typedef typename promote_type<eT1,eT2>::result out_eT;
    promote_type<eT1,eT2>::check();
    return mtGlue<out_eT, T1, T2, glue_mixed_schur>( X.get_ref(), Y.get_ref() );
  }


  //! Base * scalar
  template<typename T1>
  arma_inline
  const eOp<T1, eop_scalar_times>
  schur
  (const Base<typename T1::elem_type,T1>& X, const typename T1::elem_type k)
  {
    arma_extra_debug_sigprint();
    return eOp<T1, eop_scalar_times>(X.get_ref(),k);
  }

  //! scalar * Base
  template<typename T1>
  arma_inline
  const eOp<T1, eop_scalar_times>
  schur
  (const typename T1::elem_type k, const Base<typename T1::elem_type,T1>& X)
  {
    arma_extra_debug_sigprint();
    return eOp<T1, eop_scalar_times>(X.get_ref(),k);  // NOTE: order is swapped
  }

  const double schur(const int x, const double y) { return x * y; }
  const double schur(const double x, const int y) { return x * y; }
  const double schur(const double& x, const double& y) { return x * y; }
  const double schur(const int& x, const int& y) { return x * y; }

}

// Stochastic/Math related functions
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

  double dim_size(const arma::subview_elem2<double, arma::Mat<unsigned int>, arma::Mat<unsigned int> >& x) {
    arma::mat m(x);
    return m.n_elem;
  }

  double dim_size(const arma::subview_elem1<double, arma::Mat<unsigned int> >& x) {
    arma::mat m(x);
    return m.n_elem;
  }

  template<typename T>
  double dim_size(const T& x) {
    return x.n_elem;
  }

  bool any(const bool x) {
    return x;
  }

  bool any(const arma::umat& x) {
    const arma::umat ans(arma::find(x,1));
    return ans.n_elem > 0;
  }

  static inline double square(double x) {
    return x*x;
  }

  static inline int square(int x) {
    return x*x;
  }

  double mahalanobis(const arma::vec& x, const arma::vec& mu, const arma::mat& sigma) {
    const arma::vec err = x - mu;
    return arma::as_scalar(err.t() * sigma.i() * err);
  }

  double mahalanobis(const arma::rowvec& x, const arma::rowvec& mu, const arma::mat& sigma) {
    const arma::rowvec err = x - mu;
    return arma::as_scalar(err * sigma.i() * err.t());
  }

  template<typename T, typename U, typename V>
  double normal_logp(const T& x, const U& mu, const V& tau) {
    return arma::accu(0.5*log_approx(0.5*tau/arma::math::pi()) - 0.5 * arma::schur(tau, square(x - mu)));
  }

  template<typename T, typename U, typename V>
  double uniform_logp(const T& x, const U& lower, const V& upper) {
    return (any(x < lower) || any(x > upper)) ? -std::numeric_limits<double>::infinity() : -arma::accu(log_approx(upper - lower));
  }

  template<typename T, typename U, typename V>
  double gamma_logp(const T& x, const U& alpha, const V& beta) {
    return any(x < 0 ) ?
      -std::numeric_limits<double>::infinity() :
      arma::accu(arma::schur((alpha - 1.0),log_approx(x)) - arma::schur(beta,x) - lgamma(alpha) + arma::schur(alpha,log_approx(beta)));
  }

  template<typename T, typename U, typename V>
  double beta_logp(const T& x, const U& alpha, const V& beta) {
    const double one = 1.0;
    return any(x <= 0 ) || any(x >= 1 ) || any(alpha <= 0) || any(beta <= 0) ?
      -std::numeric_limits<double>::infinity() :
      arma::accu(lgamma(alpha+beta) - lgamma(alpha) - lgamma(beta) + arma::schur((alpha-one),log_approx(x)) + arma::schur((beta-one),log_approx(one-x)));
  }

  double categorical_logp(const arma::ivec& x, const arma::mat& p) {
    if(any(p <= 0) || any(p >= 1) || any(x < 0) || any(x >= p.n_cols)) {
      return -std::numeric_limits<double>::infinity();
    }
    // replace w/ call to p.elems later
    double ans(0);
    for(unsigned int i = 0; i < x.n_rows; i++) {
      ans += log_approx(p(i,x[i]));
    }
    return ans;
  }

  double categorical_logp(const arma::ivec& x, const arma::vec& p) {
    if(any(p <= 0) || any(p >= 1) || any(x < 0) || any(x >= p.n_elem)) {
      return -std::numeric_limits<double>::infinity();
    }
    // replace w/ call to p.elems later
    double ans(0);
    for(unsigned int i = 0; i < x.n_rows; i++) {
      ans += log_approx(p(x[i]));
    }
    return ans;
  }

  double categorical_logp(const int x, const arma::vec& p) {
    return log_approx(p[x]);
  }

  template<typename T, typename U, typename V>
  double binom_logp(const T& x, const U& n, const V& p) {
    if(any(p <= 0) || any(p >= 1) || any(x < 0)  || any(x > n)) {
      return -std::numeric_limits<double>::infinity();
    }
    return arma::accu(arma::schur(x,log_approx(p)) + arma::schur((n-x),log_approx(1-p)) + arma::factln(n) - arma::factln(x) - arma::factln(n-x));
  }

  template<typename T, typename U>
  double bernoulli_logp(const T& x, const U& p) {
    if( any(p <= 0 ) || any(p >= 1) || any(x < 0)  || any(x > 1) ) {
      return -std::numeric_limits<double>::infinity();
    } else {
      return arma::accu(arma::schur(x,log_approx(p)) + arma::schur((1-x), log_approx(1-p)));
    }
  }

  template<typename T, typename U>
  double exponential_logp(const T& x, const U& lambda) {
    return arma::accu(log_approx(lambda) - arma::schur(lambda, x));
  }

  // sigma denotes cov matrix rather than precision matrix
  double multivariate_normal_sigma_logp(const arma::rowvec& x, const arma::rowvec& mu, const arma::mat& sigma) {
    const double log_2pi = log(2 * arma::math::pi());
    arma::mat R(arma::zeros<arma::mat>(sigma.n_cols,sigma.n_cols));

    // non-positive definite test via chol
    if(chol(R,sigma) == false) { return -std::numeric_limits<double>::infinity(); }

    // otherwise calc logp
    return -arma::accu(x.n_elem * log_2pi + log_approx(arma::det(sigma)) + mahalanobis(x,mu,sigma))/2;
  }

  // sigma denotes cov matrix rather than precision matrix
  double multivariate_normal_sigma_logp(const arma::vec& x, const arma::vec& mu, const arma::mat& sigma) {
    const double log_2pi = log(2 * arma::math::pi());
    arma::mat R(arma::zeros<arma::mat>(sigma.n_cols,sigma.n_cols));

    // non-positive definite test via chol
    if(chol(R,sigma) == false) { return -std::numeric_limits<double>::infinity(); }

    // otherwise calc logp
    return -arma::accu(x.n_elem * log_2pi + log_approx(arma::det(sigma)) + mahalanobis(x,mu,sigma))/2;
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
#endif // MCMC_STOCHASTIC_HPP

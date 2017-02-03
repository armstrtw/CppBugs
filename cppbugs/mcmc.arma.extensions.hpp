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

#pragma once

#include <armadillo>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <cppbugs/mcmc.icsi.log.hpp>

namespace arma {

  // log_approx
  class eop_log_approx : public eop_core<eop_log_approx> {};

  template<> template<typename eT> arma_hot arma_inline eT
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

  template<> template<typename eT> arma_hot arma_inline eT
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

  // cube
  //! element-wise multiplication of BaseCube objects with same element type
  template<typename T1, typename T2>
  arma_inline
  const eGlueCube<T1, T2, eglue_schur>
  schur
  (
   const BaseCube<typename T1::elem_type,T1>& X,
   const BaseCube<typename T1::elem_type,T2>& Y
   )
  {
    arma_extra_debug_sigprint();
    return eGlueCube<T1, T2, eglue_schur>(X.get_ref(), Y.get_ref());
  }

  //! element-wise multiplication of BaseCube objects with different element types
  template<typename T1, typename T2>
  inline
  const mtGlueCube<typename promote_type<typename T1::elem_type, typename T2::elem_type>::result, T1, T2, glue_mixed_schur>
  schur
  (
   const BaseCube< typename force_different_type<typename T1::elem_type, typename T2::elem_type>::T1_result, T1>& X,
   const BaseCube< typename force_different_type<typename T1::elem_type, typename T2::elem_type>::T2_result, T2>& Y
   )
  {
    arma_extra_debug_sigprint();
    typedef typename T1::elem_type eT1;
    typedef typename T2::elem_type eT2;
    typedef typename promote_type<eT1,eT2>::result out_eT;
    promote_type<eT1,eT2>::check();
    return mtGlueCube<out_eT, T1, T2, glue_mixed_schur>( X.get_ref(), Y.get_ref() );
  }

  // matrix
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

  double schur(const int x, const double y) { return x * y; }
  double schur(const double x, const int y) { return x * y; }
  double schur(const double& x, const double& y) { return x * y; }
  double schur(const int& x, const int& y) { return x * y; }

  // insert an 'any' function for bools into the arma namespace
  bool any(const bool x) {
    return x;
  }

  bool vectorise(bool x) {
    return x;
  }
} // namespace arma


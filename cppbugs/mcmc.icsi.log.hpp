////////////////////////////////////////////////////////////////////////////////////
// ICSIlog Copyright taken from ICSIlog source                                    //
// Copyright (C) 2007 International Computer Science Institute                    //
// 1947 Center Street, Suite 600                                                  //
// Berkeley, CA 94704                                                             //
//                                                                                //
// Contact information:                                                           //
//    Oriol Vinyals	vinyals@icsi.berkeley.edu                                 //
//    Gerald Friedland 	fractor@icsi.berkeley.edu                                 //
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
// Oriol Vinyals	vinyals@icsi.berkeley.edu                                 //
// Gerald Friedland 	fractor@icsi.berkeley.edu                                 //
//                                                                                //
// Acknowledgements                                                               //
// ----------------                                                               //
//                                                                                //
// Thanks to Harrison Ainsworth (hxa7241@gmail.com) for his idea that             //
// doubled the accuracy.                                                          //
////////////////////////////////////////////////////////////////////////////////////

#ifndef MCMC_ICSI_LOG_HPP
#define MCMC_ICSI_LOG_HPP

namespace cppbugs {


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

} // namespace cppbugs

#endif // MCMC_ICSI_LOG_HPP

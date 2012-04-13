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

#ifndef MCMC_SPECIALIZED_HPP
#define MCMC_SPECIALIZED_HPP

#include <list>
#include <armadillo>
#include <cppbugs/mcmc.object.hpp>

namespace cppbugs {

  template<typename T>
  class MCMCSpecialized : public MCMCObject {
  protected:
    bool save_history_;
  public:
    std::list<T> history;
    MCMCSpecialized(): MCMCObject(), save_history_(true) {}

    static void fill(arma::ivec& x) { x.fill(0); }
    static void fill(arma::mat& x) { x.fill(0); }
    static void fill(double& x) { x = 0; }
    static void fill(int& x) { x = 0; }

    T mean() const {
      if(history.size() == 0) {
        return T();
      }

      T ans(*history.begin());
      fill(ans);
      for(typename std::list<T>::const_iterator it = history.begin(); it != history.end(); it++) {
        ans += *it;
      }
      ans /= static_cast<double>(history.size());
      return ans;
    }
    void setSaveHistory(const bool save_history) {
      save_history_ = save_history;
    }
  };

} // namespace cppbugs
#endif //MCMC_SPECIALIZED_HPP

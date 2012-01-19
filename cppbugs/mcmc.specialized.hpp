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
#include <cppbugs/mcmc.object.hpp>

namespace cppbugs {

  template<typename T>
  class MCMCSpecialized : public MCMCObject {
    bool save_history_;
  public:
    T& value;
    T old_value;
    std::list<T> history;
    MCMCSpecialized(T& shape): MCMCObject(), save_history_(true), value(shape), old_value(shape) {}

    static int sum_dims(const double& value) { return 1; }
    static int sum_dims(const arma::mat& value) { return value.n_elem; }
    static int sum_dims(const arma::ivec& value) { return value.n_elem; }
    static void fill(arma::mat& value) { value.fill(0); }
    static void fill(double& value) { value = 0; }

    void preserve() { old_value = value; }
    void revert() { value = old_value; }
    void tally() { if(save_history_) { history.push_back(value); } }
    void print() const { std::cout << value << std::endl; }
    T mean() const {
      T ans(value);
      fill(ans);
      for(typename std::list<T>::const_iterator it = history.begin(); it != history.end(); it++) {
        ans += *it;
      }
      ans /= static_cast<double>(history.size());
      return ans;
    }
    int getSize() const { return sum_dims(value); }
    void setScale(const double scale) {}
    void setSaveHistory(const bool save_history) {
      save_history_ = save_history;
    }

    MCMCSpecialized<T>& operator=(const T& rvalue) {
      value = rvalue;
      return *this;
    }
  };

} // namespace cppbugs
#endif //MCMC_SPECIALIZED_HPP

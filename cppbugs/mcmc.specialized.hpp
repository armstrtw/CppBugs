///////////////////////////////////////////////////////////////////////////
// Copyright (C) 2010 Whit Armstrong                                     //
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
  public:
    T value;
    T old_value;
    std::list<T> history;
    MCMCSpecialized(const T& shape): MCMCObject(), value(shape), old_value(shape) {}
    void preserve() { old_value = value; }
    void revert() { value = old_value; }
    void initHistory(const size_t i) {}
    void tally(const size_t i) { history.push_back(value); }
    void print() const { std::cout << value << std::endl; }
    T mean() const {
      T ans(value);
      ans.fill(0);
      for(typename std::list<T>::const_iterator it = history.begin(); it != history.end(); it++) {
        ans += *it;
      }
      ans /= static_cast<double>(history.size());
      return ans;
    }
    int getSize() const { return static_cast<int>(value.n_elem); }
    void setScale(const double scale) {}
  };

  template<>
  class MCMCSpecialized<double> : public MCMCObject {
  public:
    double value;
    double old_value;
    std::list<double> history;
    MCMCSpecialized(const double shape): MCMCObject(), value(shape), old_value(shape) {}
    void preserve() { old_value = value; }
    void revert() { value = old_value; }
    void initHistory(const size_t i) {}
    void tally(const size_t i) { history.push_back(value); }
    void print() const { std::cout << value << std::endl; }
    double mean() const {
      double ans(0);
      for(typename std::list<double>::const_iterator it = history.begin(); it != history.end(); it++) {
        ans += *it;
      }
      ans /= static_cast<double>(history.size());
      return ans;
    }
    int getSize() const { return 1; }
    void setScale(const double scale) {}
  };

} // namespace cppbugs
#endif //MCMC_SPECIALIZED_HPP

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

#pragma once


namespace cppbugs {

  class MCMCTracked {
  public:
    virtual void track() = 0;
  };

  template<typename T, template<typename U, class Alloc = std::allocator<U> > class CONTAINER>
  class MCMCTrackedT : public MCMCTracked {
    const T& value_;
  public:
    CONTAINER<T> history;
    MCMCTrackedT(const T& value): value_(value) {}
    void track() { history.push_back(value_); }
  };
} // namespace cppbugs

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

#pragma once

#include <armadillo>
#include <cppbugs/mcmc.dynamic.stochastic.hpp>
#include <cppbugs/mcmc.observed.hpp>

namespace cppbugs {

  template<typename T, typename U, typename V, typename W, typename X>
  class MvCar : public DynamicStochastic<T> {
  private:
    const U& adj_;
    const V& weight_;
    const W& numNeigh_;
    const X& tau_;
  public:
    // b1[1:2,1:X]  ~ mv.car(adj_b1[], weight_b1[], numNeigh_b1[], tau_b1[,] )
    MvCar(T& value, const U& adj, const V& weight, const W& numNeigh, const X& tau):
      DynamicStochastic<T>(value), adj_(adj), weight_(weight), numNeigh_(numNeigh), tau_(tau) {

      if(value.n_elem != adj_.n_elem || adj_.n_elem != weight_.n_elem || weight_.n_elem != numNeigh_.n_elem || value.n_elem != tau_.n_rows) {
        throw std::logic_error("MvCar: dims do not match.");
      }

      if(tau_.n_rows != tau_.n_cols) {
        throw std::logic_error("MvCar: tau is not diagonal.");
      }
    }

    // modified jumper to preserve mv car constraints
    void jump(RngBase& rng) {
    }

    ~MvCar() {}
    const double loglik() const { return mvcar_logp(DynamicStochastic<T>::value,adj_,weight_,numNeigh_,tau_); }
  };

} // namespace cppbugs

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

#ifndef MCMC_MODEL_HPP
#define MCMC_MODEL_HPP

#include <cmath>
#include <iostream>
#include <vector>
#include <map>
#include <exception>
#include <boost/random.hpp>
#include <cppbugs/mcmc.rng.hpp>
#include <cppbugs/mcmc.object.hpp>
#include <cppbugs/mcmc.stochastic.hpp>
#include <cppbugs/mcmc.observed.hpp>
#include <cppbugs/mcmc.tracked.hpp>
#include <cppbugs/mcmc.gcc.version.hpp>

namespace cppbugs {
  typedef std::map<void*,MCMCObject*> vmc_map;
  typedef std::map<void*,MCMCObject*>::iterator vmc_map_iter;

  template<class RNG>
  class MCModel {
  private:
    double accepted_,rejected_,logp_value_,old_logp_value_;
    SpecializedRng<RNG> rng_;
    std::vector<MCMCObject*> mcmcObjects, jumping_nodes, dynamic_nodes;
    std::vector<Stochastic*> stochastic_nodes;
    std::vector<MCMCTracked*> tracked_nodes;
    std::function<void ()> update;
    vmc_map data_node_map;

    void jump() { for(auto v : jumping_nodes) { v->jump(rng_); } }
    void preserve() { for(auto v : dynamic_nodes) { v->preserve(); } }
    void revert() { for(auto v : dynamic_nodes) { v->revert(); } }
    void set_scale(const double scale) { for(auto v : jumping_nodes) { v->setScale(scale); } }
    void tally() { for(auto v : tracked_nodes) { v->track(); } }
    static bool bad_logp(const double value) { return std::isnan(value) || value == -std::numeric_limits<double>::infinity() ? true : false; }
  public:
    MCModel(std::function<void ()> update_): accepted_(0), rejected_(0), logp_value_(-std::numeric_limits<double>::infinity()), old_logp_value_(-std::numeric_limits<double>::infinity()), update(update_) {}
    ~MCModel() {
      // use data_node_map as delete list
      // only objects allocated by this class are inserted thre
      // addNode allows user allocated objects to enter the mcmcObjects vector
      for(auto m : data_node_map) {
        delete m.second;
      }
    }

    double acceptance_ratio() const {
      return accepted_ / (accepted_ + rejected_);
    }

    bool reject(const double value, const double old_logp) {
      return bad_logp(value) || log(rng_.uniform()) > (value - old_logp) ? true : false;
    }

    const double logp() const {
      double ans(0);
      for(auto node : stochastic_nodes) {
        ans += node->loglik();
      }
      return ans;
    }

    void resetAcceptanceRatio() {
      accepted_ = 0;
      rejected_ = 0;
    }

    void tune(int iterations, int tuning_step) {
      double logp_value,old_logp_value;
      logp_value  = -std::numeric_limits<double>::infinity();
      old_logp_value = -std::numeric_limits<double>::infinity();

      for(int i = 1; i <= iterations; i++) {
	for(auto it : jumping_nodes) {
          old_logp_value = logp_value;
          it->preserve();
          it->jump(rng_);
          update();
          logp_value = logp();
          if(reject(logp_value, old_logp_value)) {
            it->revert();
            logp_value = old_logp_value;
            it->reject();
          } else {
            it->accept();
          }
	}
	if(i % tuning_step == 0) {
          //std::cout << "tuning at step: " << i << std::endl;
	  for(auto it : jumping_nodes) {
	    it->tune();
	  }
	}
      }
    }

    void step() {
      old_logp_value_ = logp_value_;
      preserve();
      jump();
      update();
      logp_value_ = logp();
      if(reject(logp_value_, old_logp_value_)) {
        revert();
        logp_value_ = old_logp_value_;
        rejected_ += 1;
      } else {
        accepted_ += 1;
      }
    }

    void tune_global(int iterations, int tuning_step) {
      const double thresh = 0.1;
      // FIXME: this should possibly related to the overall size/dimension
      // of the parmaeters to be estimtated, as there is somewhat of a leverage effect
      // via the number of parameters
      const double dilution = 0.10;
      double total_size = 0;

      for(size_t i = 0; i < dynamic_nodes.size(); i++) {
        if(dynamic_cast<Stochastic*>(dynamic_nodes[i])) {
          total_size += dynamic_nodes[i]->size();
        }
      }
      double target_ar = std::max(1/log2(total_size + 3), 0.234);
      for(int i = 1; i <= iterations; i++) {
        step();
        if(i % tuning_step == 0) {
          double diff = acceptance_ratio() - target_ar;
          resetAcceptanceRatio();
          if(std::abs(diff) > thresh) {
            double adj_factor = (1.0 + diff * dilution);
            for(size_t i = 0; i < dynamic_nodes.size(); i++) {
              dynamic_nodes[i]->setScale(dynamic_nodes[i]->getScale() * adj_factor);
            }
          }
        }
      }
    }

    void burn(int iterations) {
      for(int i = 0; i < iterations; i++) {
        step();
      }
    }

    void sample(int iterations, int thin) {
      for(int i = 1; i <= iterations; i++) {
        step();
        if(i % thin == 0) { tally(); }
      }
    }

    // push into specific lists here
    // b/c we can use this cast:
    // if(dynamic_cast<Observed<T>* >(node))
    // as a proxy for the old isObserved() function
    template<template<typename,typename,typename> class MCTYPE, typename T, typename U, typename V>
    MCTYPE<T, U, V>& link(T& x, const U& a, const V& b) {
      MCTYPE<T, U, V>* node = new MCTYPE<T, U, V>(x, a, b);

      // test object for traits
      Stochastic* sp = dynamic_cast<Stochastic*>(node);
      Observed<T>* op = dynamic_cast<Observed<T>* >(node);
      Dynamic<T>* dp = dynamic_cast<Dynamic<T>* >(node);

      if(sp) {
        stochastic_nodes.push_back(node);
        if(node->loglik()==-std::numeric_limits<double>::infinity()) {
          // throw
        }
      }

      // only jump stochastics which are not observed
      if(sp && op == NULL) jumping_nodes.push_back(node);
      if(dp) dynamic_nodes.push_back(node);

      data_node_map[(void*)(&x)] = node;

      return *node;
    }

    template<template<typename,typename,typename> class MCTYPE, typename T, typename U, typename V>
    MCTYPE<T, U, V>& link(const T& x, const U& a, const V& b) {
      MCTYPE<T, U, V>* node = new MCTYPE<T, U, V>(x, a, b);

      // test object for traits
      Stochastic* sp = dynamic_cast<Stochastic*>(node);
      Observed<T>* op = dynamic_cast<Observed<T>*>(node);
      Dynamic<T>* dp = dynamic_cast<Dynamic<T>*>(node);

      if(sp) {
        stochastic_nodes.push_back(node);
        if(sp->loglik()==-std::numeric_limits<double>::infinity()) {
          // throw
        }
      }

      // only jump stochastics which are not observed
      if(sp && op == NULL) jumping_nodes.push_back(node);
      if(dp) dynamic_nodes.push_back(node);

      data_node_map[(void*)(&x)] = node;

      return *node;
    }
#if GCC_VERSION > 40700
    template<template<typename,typename,typename> class MCTYPE, typename T, typename U, typename V>
    MCTYPE<T, U, V>& link(T& x, const U&& a, const V& b) {
      MCTYPE<T, U, V>* node = new MCTYPE<T, U, V>(x, std::move(a), b);

      // test object for traits
      Stochastic* sp = dynamic_cast<Stochastic*>(node);
      Observed<T>* op = dynamic_cast<Observed<T>* >(node);
      Dynamic<T>* dp = dynamic_cast<Dynamic<T>* >(node);

      if(sp) {
        stochastic_nodes.push_back(node);
        if(node->loglik()==-std::numeric_limits<double>::infinity()) {
          // throw
        }
      }

      // only jump stochastics which are not observed
      if(sp && op == NULL) jumping_nodes.push_back(node);
      if(dp) dynamic_nodes.push_back(node);

      data_node_map[(void*)(&x)] = node;

      return *node;
    }

    template<template<typename,typename,typename> class MCTYPE, typename T, typename U, typename V>
    MCTYPE<T, U, V>& link(T& x, const U& a, const V&& b) {
      MCTYPE<T, U, V>* node = new MCTYPE<T, U, V>(x, a, std::move(b));

      // test object for traits
      Stochastic* sp = dynamic_cast<Stochastic*>(node);
      Observed<T>* op = dynamic_cast<Observed<T>* >(node);
      Dynamic<T>* dp = dynamic_cast<Dynamic<T>* >(node);

      if(sp) {
        stochastic_nodes.push_back(node);
        if(node->loglik()==-std::numeric_limits<double>::infinity()) {
          // throw
        }
      }

      // only jump stochastics which are not observed
      if(sp && op == NULL) jumping_nodes.push_back(node);
      if(dp) dynamic_nodes.push_back(node);

      data_node_map[(void*)(&x)] = node;

      return *node;
    }

    template<template<typename,typename,typename> class MCTYPE, typename T, typename U, typename V>
    MCTYPE<T, U, V>& link(T& x, const U&& a, const V&& b) {
      MCTYPE<T, U, V>* node = new MCTYPE<T, U, V>(x, std::move(a), std::move(b));

      // test object for traits
      Stochastic* sp = dynamic_cast<Stochastic*>(node);
      Observed<T>* op = dynamic_cast<Observed<T>* >(node);
      Dynamic<T>* dp = dynamic_cast<Dynamic<T>* >(node);

      if(sp) {
        stochastic_nodes.push_back(node);
        if(node->loglik()==-std::numeric_limits<double>::infinity()) {
          // throw
        }
      }

      // only jump stochastics which are not observed
      if(sp && op == NULL) jumping_nodes.push_back(node);
      if(dp) dynamic_nodes.push_back(node);

      data_node_map[(void*)(&x)] = node;

      return *node;
    }

    template<template<typename,typename,typename> class MCTYPE, typename T, typename U, typename V>
    MCTYPE<T, U, V>& link(const T& x, const U&& a, const V& b) {
      MCTYPE<T, U, V>* node = new MCTYPE<T, U, V>(x, std::move(a), b);

      // test object for traits
      Stochastic* sp = dynamic_cast<Stochastic*>(node);
      Observed<T>* op = dynamic_cast<Observed<T>*>(node);
      Dynamic<T>* dp = dynamic_cast<Dynamic<T>*>(node);

      if(sp) {
        stochastic_nodes.push_back(node);
        if(sp->loglik()==-std::numeric_limits<double>::infinity()) {
          // throw
        }
      }

      // only jump stochastics which are not observed
      if(sp && op == NULL) jumping_nodes.push_back(node);
      if(dp) dynamic_nodes.push_back(node);

      data_node_map[(void*)(&x)] = node;

      return *node;
    }

    template<template<typename,typename,typename> class MCTYPE, typename T, typename U, typename V>
    MCTYPE<T, U, V>& link(const T& x, const U& a, const V&& b) {
      MCTYPE<T, U, V>* node = new MCTYPE<T, U, V>(x, a, std::move(b));

      // test object for traits
      Stochastic* sp = dynamic_cast<Stochastic*>(node);
      Observed<T>* op = dynamic_cast<Observed<T>*>(node);
      Dynamic<T>* dp = dynamic_cast<Dynamic<T>*>(node);

      if(sp) {
        stochastic_nodes.push_back(node);
        if(sp->loglik()==-std::numeric_limits<double>::infinity()) {
          // throw
        }
      }

      // only jump stochastics which are not observed
      if(sp && op == NULL) jumping_nodes.push_back(node);
      if(dp) dynamic_nodes.push_back(node);

      data_node_map[(void*)(&x)] = node;

      return *node;
    }

    template<template<typename,typename,typename> class MCTYPE, typename T, typename U, typename V>
    MCTYPE<T, U, V>& link(const T& x, const U&& a, const V&& b) {
      MCTYPE<T, U, V>* node = new MCTYPE<T, U, V>(x, std::move(a), std::move(b));

      // test object for traits
      Stochastic* sp = dynamic_cast<Stochastic*>(node);
      Observed<T>* op = dynamic_cast<Observed<T>*>(node);
      Dynamic<T>* dp = dynamic_cast<Dynamic<T>*>(node);

      if(sp) {
        stochastic_nodes.push_back(node);
        if(sp->loglik()==-std::numeric_limits<double>::infinity()) {
          // throw
        }
      }

      // only jump stochastics which are not observed
      if(sp && op == NULL) jumping_nodes.push_back(node);
      if(dp) dynamic_nodes.push_back(node);

      data_node_map[(void*)(&x)] = node;

      return *node;
    }

#endif

    // this is for deterministic nodes
    template<template<typename> class MCTYPE, typename T>
    MCTYPE<T>& link(T& x) {
      MCTYPE<T>* node = new MCTYPE<T>(x);

      // test object for traits
      Dynamic<T>* dp = dynamic_cast<Dynamic<T>* >(node);
      // only jump stochastics which are not observed
      if(dp) dynamic_nodes.push_back(node);
      data_node_map[(void*)(&x)] = node;

      return *node;
    }

    template<template<typename U,class Alloc = std::allocator<U> > class CONTAINER, typename T>
    CONTAINER<T>& track(const T& x) {
      MCMCTrackedT<T,CONTAINER>* node = new MCMCTrackedT<T,CONTAINER>(x);
      tracked_nodes.push_back(node);
      return node->history;
    }

    // // allows node to be added without being put on the delete list
    // // for those who want full control of their memory...
    // void track(MCMCObject* node) {
    //   mcmcObjects.push_back(node);
    // }
  };
} // namespace cppbugs
#endif // MCMC_MODEL_HPP

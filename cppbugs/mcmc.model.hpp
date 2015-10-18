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

#include <cmath>
#include <iostream>
#include <vector>
#include <map>
#include <functional>
#include <exception>
#include <boost/random.hpp>
#include <cppbugs/mcmc.rng.base.hpp>
#include <cppbugs/mcmc.object.hpp>
#include <cppbugs/mcmc.stochastic.hpp>
#include <cppbugs/mcmc.observed.hpp>
#include <cppbugs/mcmc.deterministic.hpp>
#include <cppbugs/mcmc.tracked.hpp>
#include <cppbugs/mcmc.gcc.version.hpp>
#include <cppbugs/deterministics/mcmc.lambda.hpp>

namespace cppbugs {

  class MCModel {
  private:
    RngBase& rng_;
    double accepted_,rejected_,logp_value_,old_logp_value_;
    std::vector<MCMCObject*> mcmcObjects, jumping_nodes, dynamic_nodes, deterministic_nodes;
    std::vector<Stochastic*> stochastic_nodes;
    std::vector<MCMCTracked*> tracked_nodes;

    //void jump() { for(auto v : jumping_nodes) { v->jump(rng_); } }
    void jump() { for(auto v : dynamic_nodes) { v->jump(rng_); } }
    void jump_detrministics() { for(size_t i = 0; i < deterministic_nodes.size(); i++) { deterministic_nodes[i]->jump(rng_); } }
    void preserve() { for(auto v : dynamic_nodes) { v->preserve(); } }
    void revert() { for(auto v : dynamic_nodes) { v->revert(); } }
    void set_scale(const double scale) { for(auto v : jumping_nodes) { v->setScale(scale); } }
    void tally() { for(auto v : tracked_nodes) { v->track(); } }
    static bool bad_logp(const double value) { return std::isnan(value) || value == -std::numeric_limits<double>::infinity() ? true : false; }
  public:
    MCModel(RngBase& rng): rng_(rng), accepted_(0), rejected_(0), logp_value_(-std::numeric_limits<double>::infinity()), old_logp_value_(-std::numeric_limits<double>::infinity()) {}
    ~MCModel() {
      // only objects allocated by this class are inserted thre
      // addNode allows user allocated objects to enter the mcmcObjects vector
      for(auto m : mcmcObjects) {
        delete m;
      }
    }

    double acceptance_ratio() const {
      return accepted_ / (accepted_ + rejected_);
    }

    bool reject(const double value, const double old_logp) {
      return bad_logp(value) || log(rng_.uniform()) > (value - old_logp) ? true : false;
    }

    double logp() const {
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
          // has to be done after each stoch jump
          jump_detrministics();
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

    template<typename T>
    void addNode(MCMCObject* node) {
      mcmcObjects.push_back(node);
      // test object for traits
      Stochastic* sp = dynamic_cast<Stochastic*>(node);
      Observed<T>* op = dynamic_cast<Observed<T>* >(node);
      Dynamic<T>* dp = dynamic_cast<Dynamic<T>* >(node);
      Deterministic<T>* detp = dynamic_cast<Deterministic<T>* >(node);      

      if(sp) {
        stochastic_nodes.push_back(sp);
        if(sp->loglik()==-std::numeric_limits<double>::infinity()) {
          throw std::logic_error("Cannot start from -Inf.");
        }
      }

      // only jump stochastics which are not observed
      if(sp && op == NULL) jumping_nodes.push_back(node);
      if(dp) dynamic_nodes.push_back(node);
      if(detp) deterministic_nodes.push_back(detp);
    }

    template<typename T, typename U>
    Lambda1<T, U>& lambda(T& x, std::function<const T(const U&)> f, const U& a) {
      Lambda1<T, U>* node = new Lambda1<T, U>(x, f, a);
      addNode<T>(node);
      return *node;
    }

    template<typename T, typename U, typename V>
    Lambda2<T, U, V>& lambda(T& x, std::function<const T(const U&,const V&)> f, const U& a, const V& b) {
      Lambda2<T, U, V>* node = new Lambda2<T, U, V>(x, f, a, b);
      addNode<T>(node);
      return *node;
    }

    template<typename T, typename U, typename V, typename W>
    Lambda3<T, U, V, W>& lambda(T& x, std::function<const T(const U&,const V&,const W&)> f, const U& a, const V& b, const W& c) {
      Lambda3<T, U, V, W>* node = new Lambda3<T, U, V, W>(x, f, a, b, c);
      addNode<T>(node);
      return *node;
    }

    template<typename T, typename U, typename V, typename W, typename X>
    Lambda4<T, U, V, W, X>& lambda(T& x, std::function<const T(const U&,const V&,const W&, const X&)> f, const U& a, const V& b, const W& c, const X& d) {
      Lambda4<T, U, V, W, X>* node = new Lambda4<T, U, V, W, X>(x, f, a, b, c, d);
      addNode<T>(node);
      return *node;
    }

    template<template<typename,typename> class MCTYPE, typename T, typename U>
    MCTYPE<T, U>& link(T& x, const U& a) {
      MCTYPE<T, U>* node = new MCTYPE<T, U>(x, a);
      addNode<T>(node);
      return *node;
    }

    template<template<typename,typename,typename> class MCTYPE, typename T, typename U, typename V>
    MCTYPE<T, U, V>& link(T& x, const U& a, const V& b) {
      MCTYPE<T, U, V>* node = new MCTYPE<T, U, V>(x, a, b);
      addNode<T>(node);
      return *node;
    }

    template<template<typename,typename,typename,typename> class MCTYPE, typename T, typename U, typename V, typename W>
    MCTYPE<T, U, V, W>& link(T& x, const U& a, const V& b, const W& c) {
      MCTYPE<T, U, V, W>* node = new MCTYPE<T, U, V, W>(x, a, b, c);
      addNode<T>(node);
      return *node;
    }

    template<template<typename,typename,typename,typename,typename> class MCTYPE, typename T, typename U, typename V, typename W, typename X>
    MCTYPE<T, U, V, W, X>& link(T& x, const U& a, const V& b, const W& c, const X& d) {
      MCTYPE<T, U, V, W, X>* node = new MCTYPE<T, U, V, W, X>(x, a, b, c, d);
      addNode<T>(node);
      return *node;
    }

    template<template<typename,typename,typename> class MCTYPE, typename T, typename U, typename V>
    MCTYPE<T, U, V>& link(const T& x, const U& a, const V& b) {
      MCTYPE<T, U, V>* node = new MCTYPE<T, U, V>(x, a, b);
      addNode<T>(node);
      return *node;
    }

#if GCC_VERSION > 40700
    template<template<typename,typename,typename> class MCTYPE, typename T, typename U, typename V>
    MCTYPE<T, U, V>& link(T& x, const U&& a, const V& b) {
      MCTYPE<T, U, V>* node = new MCTYPE<T, U, V>(x, std::move(a), b);
      addNode<T>(node);
      return *node;
    }

    template<template<typename,typename,typename> class MCTYPE, typename T, typename U, typename V>
    MCTYPE<T, U, V>& link(T& x, const U& a, const V&& b) {
      MCTYPE<T, U, V>* node = new MCTYPE<T, U, V>(x, a, std::move(b));
      addNode<T>(node);
      return *node;
    }

    template<template<typename,typename,typename> class MCTYPE, typename T, typename U, typename V>
    MCTYPE<T, U, V>& link(T& x, const U&& a, const V&& b) {
      MCTYPE<T, U, V>* node = new MCTYPE<T, U, V>(x, std::move(a), std::move(b));
      addNode<T>(node);
      return *node;
    }

    template<template<typename,typename,typename> class MCTYPE, typename T, typename U, typename V>
    MCTYPE<T, U, V>& link(const T& x, const U&& a, const V& b) {
      MCTYPE<T, U, V>* node = new MCTYPE<T, U, V>(x, std::move(a), b);
      addNode<T>(node);
      return *node;
    }

    template<template<typename,typename,typename> class MCTYPE, typename T, typename U, typename V>
    MCTYPE<T, U, V>& link(const T& x, const U& a, const V&& b) {
      MCTYPE<T, U, V>* node = new MCTYPE<T, U, V>(x, a, std::move(b));
      addNode<T>(node);
      return *node;
    }

    template<template<typename,typename,typename> class MCTYPE, typename T, typename U, typename V>
    MCTYPE<T, U, V>& link(const T& x, const U&& a, const V&& b) {
      MCTYPE<T, U, V>* node = new MCTYPE<T, U, V>(x, std::move(a), std::move(b));
      addNode<T>(node);
      return *node;
    }
#endif

    // this is for deterministic nodes
    template<template<typename> class MCTYPE, typename T>
    MCTYPE<T>& link(T& x) {
      MCTYPE<T>* node = new MCTYPE<T>(x);
      addNode<T>(node);
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

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

#ifndef MCMC_MODEL_HPP
#define MCMC_MODEL_HPP

#include <cmath>
#include <iostream>
#include <vector>
#include <boost/random.hpp>
#include <cppbugs/mcmc.rng.hpp>
#include <cppbugs/mcmc.model.base.hpp>
#include <cppbugs/mcmc.object.hpp>
#include <cppbugs/cppbugs.hpp>

namespace cppbugs {

  class MCModel : public MCModelBase {
  private:
    double accepted_;
    double rejected_;
    SpecializedRng<boost::minstd_rand> rng_;
    std::vector<MCMCObject*> mcmcObjects, jumping_stochastics, deterministics;
    std::vector<const double*> logps;
    void jump_all(std::vector<MCMCObject*>& v) { for(size_t i = 0; i < v.size(); i++) { v[i]->jump(rng_); } }
    void preserve_all(std::vector<MCMCObject*>& v) { for(size_t i = 0; i < v.size(); i++) { v[i]->preserve(); } }
    void revert_all(std::vector<MCMCObject*>& v) { for(size_t i = 0; i < v.size(); i++) { v[i]->revert(); } }
    void tally_all(std::vector<MCMCObject*>& v) { for(size_t i = 0; i < v.size(); i++) { v[i]->tally(); } }
    void print_all(std::vector<MCMCObject*>& v) { for(size_t i = 0; i < v.size(); i++) { v[i]->print(); } }
    bool bad_logp(const double value) const { return isnan(value) || value == -std::numeric_limits<double>::infinity() ? true : false; }
  public:
    MCModel(): MCModelBase(), accepted_(0), rejected_(0) {}

    void add(MCMCObject& p) {
      mcmcObjects.push_back(&p);

      if(p.isStochastic()) {
        //std::cout << *p.getLogp() << std::endl;
        logps.push_back(p.getLogp());
      }

      if(p.isStochastic() && !p.isObserved()) {
        jumping_stochastics.push_back(&p);
      }

      if(p.isDeterministc()) {
        deterministics.push_back(&p);
      }
    }

    double acceptance_ratio() const {
      return accepted_ / (accepted_ + rejected_);
    }

    void print() {
      print_all(mcmcObjects);
    }

    bool reject(const double value, const double old_logp) {
      double r = exp(value - old_logp);
      return bad_logp(value) || (rng_.uniform() > r && r < 1)  ? true : false;
    }

    double logp() const {
      double ans(0);
      for(size_t i = 0; i < logps.size(); i++) {
        ans += *logps[i];
      }
      return ans;
    }

    void tune(int iterations, int tuning_step) {
      double logp_value,old_logp_value;
      logp_value  = -std::numeric_limits<double>::infinity();
      old_logp_value = -std::numeric_limits<double>::infinity();

      for(int i = 1; i <= iterations; i++) {
	for(std::vector<MCMCObject*>::iterator it = jumping_stochastics.begin(); it != jumping_stochastics.end(); it++) {
          old_logp_value = logp_value;
          (*it)->preserve();
          (*it)->jump(rng_);
          update();
          logp_value = logp();
          if(reject(logp_value, old_logp_value)) {
            (*it)->revert();
            logp_value = old_logp_value;
            (*it)->reject();
          } else {
            (*it)->accept();
          }
	}
	if(i % tuning_step == 0) {
          //std::cout << "tuning at step: " << i << std::endl;
	  for(std::vector<MCMCObject*>::iterator it = jumping_stochastics.begin(); it != jumping_stochastics.end(); it++) {
	    (*it)->tune();
	  }
	}
      }
    }

    void sample(int iterations, int burn, int adapt, int thin) {
      double logp_value,old_logp_value;

      // tuning phase
      tune(adapt,static_cast<int>(adapt/100));

      logp_value  = -std::numeric_limits<double>::infinity();
      old_logp_value = -std::numeric_limits<double>::infinity();
      for(int i = 1; i <= (iterations + burn); i++) {
        old_logp_value = logp_value;
        preserve_all(mcmcObjects);
        jump_all(jumping_stochastics);
        update();
        logp_value = logp();
        if(reject(logp_value, old_logp_value)) {
          revert_all(mcmcObjects);
          logp_value = old_logp_value;
          rejected_ += 1;
        } else {
          accepted_ += 1;
        }
        if(i > burn && (i % thin == 0)) {
          tally_all(mcmcObjects);
        }
      }
    }
  };
} // namespace cppbugs
#endif // MCMC_MODEL_HPP

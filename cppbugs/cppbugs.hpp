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

#ifndef CPPBUGS_HPP
#define CPPBUGS_HPP

#include <cmath>
#include <iostream>
#include <vector>
#include <map>
#include <armadillo>
#include <boost/random.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <cppbugs/mcmc.rng.hpp>
#include <cppbugs/mcmc.object.hpp>
#include <cppbugs/mcmc.model.hpp>

namespace cppbugs {

  double factln_single(int n) {
    if(n > 100) {
      return boost::math::lgamma(static_cast<double>(n) + 1);
    }
    double ans(1);
    for (int i=n; i>1; i--) {
      ans *= i;
    }
    return log(ans);
  }

  double factln(const int i) {
    static std::map<int,double> factln_table;

    if(i < 0) {
      return -std::numeric_limits<double>::infinity();
    }
    std::map<int,double>::iterator it = factln_table.find(i);
    if(it == factln_table.end()) {
      double ans = factln_single(i);
      factln_table[i] = ans;
      return ans;
    } else {
      return it->second;
    }
  }

  arma::mat factln(const arma::imat& x) {
    arma::mat ans; ans.copy_size(x);
    for(size_t i = 0; i < x.n_elem; i++) {
      ans[i] = factln(x[i]);
    }
    return ans;
  }


  template<typename T>
  class MCMCSpecialized : public MCMCObject {
  public:
    T value;
    T old_value;
    std::vector<T> history;
    MCMCSpecialized(const T& shape): MCMCObject(), value(shape), old_value(shape) {}
    void preserve() { old_value = value; }
    void revert() { value = old_value; }
    void tally() { history.push_back(value); }
    void print() const { std::cout << value << std::endl; }
    T mean() const {
      T ans(value);
      ans.fill(0);
      for(size_t i = 0; i < history.size(); i++) {
        ans += history[i];
      }
      ans /= static_cast<double>(history.size());
      return ans;
    }
  };

  template<>
  class MCMCSpecialized<double> : public MCMCObject {
  public:
    double value;
    double old_value;
    std::vector<double> history;
    MCMCSpecialized(const double shape): MCMCObject(), value(shape), old_value(shape) {}
    void preserve() { old_value = value; }
    void revert() { value = old_value; }
    void tally() { history.push_back(value); }
    void print() const { std::cout << value << std::endl; }
    double mean() const {
      double ans(0);
      for(size_t i = 0; i < history.size(); i++) {
        ans += history[i];
      }
      ans /= static_cast<double>(history.size());
      return ans;
    }
  };

  template<typename T>
  class Deterministic : public MCMCSpecialized<T> {
  public:
    Deterministic(const T& value): MCMCSpecialized<T>(value) {}
    void jump(RngBase& rng) {}
    bool isDeterministc() const { return true; }
    bool isStochastic() const { return false; }
  };

  template<typename T, typename U>
  void stochastic_jump(T& value, U& rng) {
    for(size_t i = 0; i < value.n_elem; i++) {
      value[i] += rng.normal();
    }
  }

  template<typename U>
  void stochastic_jump(double& value, U& rng) {
    value += rng.normal();
  }

  template<typename T>
  class Stochastic : public MCMCSpecialized<T> {
  protected:
    bool observed_;
    Mat accepted_;
    Mat rejected_;
    Mat scale_;
  public:
    Stochastic(const T& value, const bool observed): MCMCSpecialized<T>(value), observed_(observed),
						     accepted_(conv_to<mat>::from(T)), rejected_(conv_to<mat>::from(T)),
						     scale_(T.ones()) {}
    // deterministics only derive their logp from their parents
    //double logp() const { return 0; }
    // do nothing, object must be updated after all other objects are jumped
    bool isDeterministc() const { return false; }
    bool isStochastic() const { return true; }
    void jump(RngBase& rng) {
      if(observed_) {
        return;
      } else {
        stochastic_jump(MCMCSpecialized<T>::value,rng);
      }
    }
    void component_jump(RngBase& rng, MCModel& m) {
      if(observed_) {
        return;
      } else {
	for(size_t i = 0; i < value.n_elem; i++) {
	  double old_logp = m.logp();
	  old_value[i] = value[i];
	  value[i] += rng.normal() * scale[i];
	  m.update();
	  if(reject(m.logp(), old_logp_value)) {
	    value[i] = old_value[i];
	    rejected_[i] += 1;
	  } else {
	    accepted_[i] += 1;
	  }
	}
      }
    }
  };

  double accu(const double x) {
    return x;
  }

  template<typename T>
  class Normal : public Stochastic<T> {
  public:
    Normal(const T& x, const bool observed = false): Stochastic<T>(x,observed) {}

    template<typename U, typename V>
    double logp(const U& mu, const V& tau) const {
      return accu(0.5*log(0.5*tau/arma::math::pi()) - 0.5 * tau * pow(Stochastic<T>::value - mu,2));
    }
  };

  template<typename T>
  class NormalStatic : public Stochastic<T> {
    double mu_, tau_;
  public:
    NormalStatic(const T& x, const double mu, const double tau, const bool observed = false): Stochastic<T>(x,observed), mu_(mu), tau_(tau) {}
    double logp() const {
      return accu(0.5*log(0.5*tau_/arma::math::pi()) - 0.5 * tau_ * pow(Stochastic<T>::value - mu_,2));
    }
  };

  template<typename T>
  class Uniform : public Stochastic<T> {
  public:
    Uniform(const T& x, const bool observed = false): Stochastic<T>(x,observed) {}
    double logp(const double lower, const double upper) const {
      return (Stochastic<T>::value < lower || Stochastic<T>::value > upper) ? -std::numeric_limits<double>::infinity() : -log(upper - lower);
    }
  };

  template<typename T>
  class UniformStatic : public Stochastic<T> {
    double lower_, upper_;
  public:
    UniformStatic(const T& x, const double lower, const double upper, const bool observed = false): Stochastic<T>(x,observed),lower_(lower),upper_(upper) {}
    double logp() const {
      return (Stochastic<T>::value < lower_ || Stochastic<T>::value > upper_) ? -std::numeric_limits<double>::infinity() : -log(upper_ - lower_);
    }
    void jump(RngBase& rng) {
      const T oldvalue(Stochastic<T>::value);
      if(Stochastic<T>::observed_) {
        return;
      } else {
        do {
          Stochastic<T>::value = oldvalue + rng.normal();
        } while(Stochastic<T>::value < lower_ || Stochastic<T>::value > upper_);
      }
    }
  };

  template<typename T>
  class GammaStatic : public Stochastic<T> {
    double alpha_, beta_;
  public:
    GammaStatic(const T& x, const double alpha, const double beta, const bool observed = false): Stochastic<T>(x,observed), alpha_(alpha), beta_(beta) {}
    double logp() const {
      return (Stochastic<T>::value < 0 ) ? -std::numeric_limits<double>::infinity() : accu( (alpha_ - 1.0) * log(Stochastic<T>::value) - beta_*Stochastic<T>::value - boost::math::lgamma(alpha_) + alpha_*log(beta_) );
    }
    void jump(RngBase& rng) {
      const T oldvalue(Stochastic<T>::value);
      if(Stochastic<T>::observed_) {
        return;
      } else {
        do {
          Stochastic<T>::value = oldvalue + rng.normal();
        } while(Stochastic<T>::value < 0);
      }
    }
  };

  template<typename T>
  class BinomialStatic : public Stochastic<T> {
    double n_, p_;
    double actual_logp() const {
      return accu(Stochastic<T>::value * log(p_) + (n_-Stochastic<T>::value)*log(1-p_) + factln(n_)-factln(Stochastic<T>::value)-factln(n_-Stochastic<T>::value));
    }
  public:
    BinomialStatic(const T& x, const double n, const double p, const bool observed = false): Stochastic<T>(x,observed), n_(n), p_(p) {}
    double logp() const {
      return Stochastic<T>::value < 0 ? -std::numeric_limits<double>::infinity() : actual_logp();
    }
    void jump(RngBase& rng) {
      const T oldvalue(Stochastic<T>::value);
      if(Stochastic<T>::observed_) {
        return;
      } else {
        do {
          Stochastic<T>::value = oldvalue + rng.normal();
        } while(Stochastic<T>::value < 0);
      }
    }
  };

  template<typename T>
  class Binomial : public Stochastic<T> {
  public:
    Binomial(const T& x, const bool observed = false): Stochastic<T>(x,observed) {}

    template<typename U, typename V>
    double logp(const U& n, const V& p) const {
      arma::uvec ltzero = find(Stochastic<T>::value < 0,1);
      if(ltzero.n_elem) {
        return -std::numeric_limits<double>::infinity();
      }
      arma::uvec gtn = find(Stochastic<T>::value > n,1);
      if(gtn.n_elem) {
        return -std::numeric_limits<double>::infinity();
      }
      return accu(Stochastic<T>::value % log(p) + (n-Stochastic<T>::value) % log(1-p) + factln(n)-factln(Stochastic<T>::value)-factln(n-Stochastic<T>::value));
      //std::cout << 1-p << "--" << std::endl;
      //return accu( (n-Stochastic<T>::value) % log(1-p) );
    }
  };
} // namespace cppbugs
#endif // CPPBUGS_HPP

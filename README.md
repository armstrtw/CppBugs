### Purpose

CppBugs is a c++ library designed for MCMC sampling.

* CppBugs is now depricated in favor of [stan](http://mc-stan.org).


### Features

CppBugs attempts to make writing mcmc models as painless as possible.  It incorporates features
from both WinBugs and PyMC and requires users only to implment an update method which resembles the model section of a WinBUGS script.

* CppBugs is fast.  Typically between 5x to 10x faster than equivalent WinBugs and 3x to 5x faster than PyMC models.

* Common statistical distributions are supported drawing heavily on Boost libraries.  Many more will be implemented
  to eventually be as feature complete as WinBugs/PyMC. 


### Usage

Starting with a bugs model:
```{.bug}
model {
    for (j in 1:J){
        y[j] ~ dnorm (theta[j], tau.y[j])
        theta[j] ~ dnorm (mu.theta, tau.theta)
        tau.y[j] <- pow(sigma.y[j], -2)
    }
    mu.theta ~ dnorm (0, 1.0E-6)
    tau.theta <- pow(sigma.theta, -2)
    sigma.theta ~ dunif (0, 1000)
}
```

This mode can be converted to a CppBugs model in two steps.

* define the variable space

* link each variable with its dependencies

```{.cpp}

  const int J = 8;
  const vec sigma_y({15,10,16,11,9,11,10,18});
  const vec tau_y = pow(sigma_y,-2);
  const vec y({28,  8, -3,  7, -1,  1, 18, 12});

  double mu_theta(0);
  double sigma_theta(1);
  double tau_theta = pow(sigma_theta,-2);
  vec theta = randn<vec>(J);

  BoostRng<boost::minstd_rand> rng;
  MCModel m(rng);

  // noninformative prior on mu
  m.link<Normal>(mu_theta, 0.0, 1.0E-6);

  // noninformative prior on sigma
  m.link<Uniform>(sigma_theta, 0, 1000);

  m.link<InvVariance>(tau_theta,sigma_theta);
  m.link<Normal>(theta,mu_theta,tau_theta);
  m.link<ObservedNormal>(y, theta, tau_y);

  // things to track
  std::vector<vec>& theta_hist = m.track<std::vector>(theta);

  m.tune(1e4,100);
  m.tune_global(1e4,100);
  m.burn(5e3);
  m.sample(1e4, 1);
```

Please see the test folder for more examples.

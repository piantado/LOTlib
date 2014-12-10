"""
**Hypothesis** -- superclass for hypotheses in Bayesian inference.

A Hypothesis mainly supports .compute_prior() and .compute_likelihood(data), which are called by sampling
  and search algorithms.

"""
import LOTlib
from LOTlib.Evaluation import Primitives
from LOTlib.Miscellaneous import *


class Hypothesis(object):
    """A hypothesis bundles together a value (hypothesis value) with a bunch of remembered states,
    like posterior_score, prior, likelihood.

    This class is typically a superclass of the thing you really want.

    Note:
        Temperatures: By default, a Hypothesis has a prior_temperature and likelihood_temperature. These
          are taken into account in setting the posterior_score (for computer_prior and compute_likelihood),
          in the values returned by these, AND in the stored values under self.prior and self.likelihood

    Args:
        value: The default value for the hypothesis.
        prior_temperature: Temperature used when running compute_prior.
        likelihood_temperature: Temperature used when running compute_likelihood.

    """
    def __init__(self, value=None, prior_temperature=1.0, likelihood_temperature=1.0, **kwargs):
        self.__dict__.update(kwargs)

        self.set_value(value)       # to zero out prior, likelhood, posterior_score
        self.prior, self.likelihood, self.posterior_score = [-Infinity, -Infinity, -Infinity]
        self.prior_temperature = prior_temperature
        self.likelihood_temperature = likelihood_temperature
        self.stored_likelihood = None
        # keep track of some calls (Global variable)
        global POSTERIOR_CALL_COUNTER
        POSTERIOR_CALL_COUNTER = 0

    def set_value(self, value):
        """Sets the (self.)value of this hypothesis to value."""
        self.value = value

    def __copy__(self):
        """Returns a copy of the Hypothesis object by calling copy() on self.value."""
        return Hypothesis(value=self.value.copy(), prior_temperature=self.prior_temperature,
                          likelihood_temperature=self.likelihood_temperature)

    # ========================================================================================================
    #  All instances of this must implement these:

    def compute_prior(self):
        """Compute the prior and stores it in self.prior.

        Note:
            This method must be implemented when writing subclasses of Hypothesis
            This *should* take into account prior_temperature

        """
        raise NotImplementedError

    def compute_single_likelihood(self, datum, **kwargs):
        """Compute the likelihood of a single data point datum, under this hypothesis.

        Note:
            This method must be implemented when writing subclasses of Hypothesis.
            It should NOT take into account likelihood_temperature, as this is done in compute_likelihood.

        """
        raise NotImplementedError

    # And the main likelihood function just maps compute_single_likelihood over the data
    def compute_likelihood(self, data, **kwargs):
        """Compute the likelihood of the iterable of data.

        This is typically NOT subclassed, as compute_single_likelihood is what subclasses should implement.

        Versions using decayed likelihood can be found in Hypothesis.DecayedLikelihoodHypothesis.

        """
        likelihoods = [self.compute_single_likelihood(datum, **kwargs) for datum in data]
        self.likelihood = sum(likelihoods) / self.likelihood_temperature
        self.posterior_score = self.prior + self.likelihood
        return self.likelihood

    # ========================================================================================================
    #  Methods for accessing likelihoods etc. on a big arrays of data

    def propose(self):
        """Generic proposal used by MCMC methods.

        This should return a list fb, newh, where fb is the forward-minus-backward log probability of the
        proposal, and newh is the proposal itself (of the same type as self).

        Note:
            This method must be implemented when writing subclasses of Hypothesis

        """
        raise NotImplementedError

    def compute_posterior(self, d, **kwargs):
        """Computes the posterior score by computing the prior and likelihood scores.
                
        Defaultly if the prior is -inf, we don't compute the likelihood (and "pretend" it's -Infinity).

        This saves us from computing likelihoods on hypotheses that we know are bad.

        """
        Primitives.LOCAL_PRIMITIVE_OPS = 0  # Reset this
        p = self.compute_prior()
        
        if p > -Infinity:        
            l = self.compute_likelihood(d, **kwargs)
        else:
            l = -Infinity   # This *could* be 0.0 if we wanted. Not clear what is best.

        self.posterior_score = p + l
        return [p, l]

    def update_posterior(self):
        """So we can save on space when writing this out in every hypothesis."""
        self.posterior_score = self.prior + self.likelihood

    # ========================================================================================================
    #  optional implementation --  if you do gibbs sampling you need:
    def enumerative_proposer(self):
        """Note: This method must be implemented when performing Gibbs sampling"""
        pass

    # ========================================================================================================
    #  These are just handy:
    def __str__(self):
        return str(self.value)
    def __repr__(self):
        return str(self)

    # for hashing hypotheses
    def __hash__(self):
        return hash(self.value)
    def __cmp__(self, x):
        return cmp(self.value, x)

    # this is for heapq algorithm in FiniteSample, which uses <= instead of cmp
    # since python implements a "min heap" we can compar elog probs
    def __le__(self, x):
        return self.posterior_score <= x.posterior_score
    def __eq__(self, other):
        return self.value.__eq__(other.value)
    def __ne__(self, other):
        return self.value.__ne__(other.value)

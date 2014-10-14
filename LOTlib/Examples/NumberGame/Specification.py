__author__ = 'eric'

from math import log
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Evaluation.Eval import *


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~ Domain-specific hypothesis wrapper class ~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
class NumberSetHypothesis(LOTHypothesis):

    def __init__(self, grammar, alpha=0.9, **kwargs):
        LOTHypothesis.__init__(self, grammar, args=[], **kwargs)
        self.alpha = alpha

    def compute_likelihood(self, data):
        """
            Likelihood of specified data being produced by this hypothesis. If datum item
            not in set, it still has 'noise' likelihood of being generated.
        """
        h = self.__call__()     # Get hypothesis set
        alpha = self.alpha
        noise = (1-alpha) / len(h)
        likelihood = 0
        for datum in data:
            if datum in h:
                likelihood += log(alpha/len(h) + noise)
            else:
                likelihood += log(noise)

        self.likelihood = likelihood
        self.posterior_score = self.prior + likelihood      # required in all compute_likelihoods
        return likelihood


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Expressions in this class evaluate like a function, e.g. 2^n + 1                      #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
class NumberExpressionHypothesis(LOTHypothesis):

    def __init__(self, grammar, domain=100, noise=0.9, args=['n'], **kwargs):
        LOTHypothesis.__init__(self, grammar, args=args, **kwargs)
        self.domain = domain
        self.noise = noise

    def compute_likelihood(self, data):
        """
            Likelihood of specified data being produced by this hypothesis. If datum item
            not in set, it still has 'noise' likelihood of being generated.
        """
        # Get subset of range [1,domain] mapped to by hypothesis function
        h = map(self, map(float, range(1, self.domain + 1)))
        h = [item for item in h if item <= self.domain]
        alpha = self.alpha
        noise = (1-alpha) / self.domain
        self.likelihood = 0
        for datum in data:
            if datum in h:
                self.likelihood += log(alpha/len(h) + noise)
            else:
                self.likelihood += log(noise)

        # This is required in all compute_likelihoods
        self.posterior_score = self.prior + self.likelihood
        return self.likelihood
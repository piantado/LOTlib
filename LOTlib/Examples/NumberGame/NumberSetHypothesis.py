__author__ = 'eric'

from math import log
from LOTlib.Hypotheses.LOTHypothesis import *
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
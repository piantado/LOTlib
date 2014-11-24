
from math import log
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
# from LOTlib.Miscellaneous import logplusexp, logsumexp, log1mexp, gammaln, Infinity
# from LOTlib.Evaluation.Eval import *


#=============================================================================================================
# Domain-specific hypothesis wrapper class
#=============================================================================================================

class NumberGameHypothesis(LOTHypothesis):
    """Wrapper class for hypotheses in the number game.

    Hypotheses evaluate to a set of numbers.

    """
    def __init__(self, grammar, alpha=0.9, domain=100, **kwargs):
        LOTHypothesis.__init__(self, grammar, args=[], **kwargs)
        self.alpha = alpha
        self.domain = domain

    def compute_likelihood(self, data, **kwargs):
        """Likelihood of specified data being produced by this hypothesis.

        If datum item not in set, it still has (1 - alpha) likelihood of being generated.

        """
        s = self()     # set of numbers corresponding to this hypothesis
                       # NOTE: This may be None if the hypothesis has too many nodes
        if isinstance(s, list):
            s = [item for item in s if item <= self.domain]
        error_p = (1.-self.alpha) / self.domain

        def compute_single_likelihood(datum):
            if s is not None and datum in s:
                likelihood = log(self.alpha/len(s) + error_p)
            else:
                likelihood = log(error_p)
            return likelihood

        likelihoods = map(compute_single_likelihood, data)
        self.likelihood = sum(likelihoods) / self.likelihood_temperature
        self.update_posterior()
        return self.likelihood

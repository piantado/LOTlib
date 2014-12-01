
from math import log
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
# from LOTlib.Miscellaneous import logplusexp, logsumexp, log1mexp, gammaln, Infinity
# from LOTlib.Evaluation.Eval import *


# ============================================================================================================
#  Domain-specific hypothesis wrapper class
# ============================================================================================================

class NumberGameHypothesis(LOTHypothesis):
    """Wrapper class for hypotheses in the number game.

    Hypotheses evaluate to a set of numbers.

    """
    def __init__(self, grammar, alpha=0.9, domain=100, **kwargs):
        LOTHypothesis.__init__(self, grammar, args=[], **kwargs)
        self.alpha = alpha
        self.domain = domain
        self.value_set = None

    def compute_likelihood(self, data, updateflag=True, **kwargs):
        """Likelihood of specified data being produced by this hypothesis.

        If datum item not in set, it still has (1 - alpha) likelihood of being generated.

        """
        s = self()      # set of numbers corresponding to this hypothesis
                        # NOTE: This may be None if the hypothesis has too many nodes
        error_p = (1.-self.alpha) / self.domain

        def compute_single_likelihood(datum, updateflag=True):
            if s is not None and datum in s:
                likelihood = log(self.alpha/len(s) + error_p)
            else:
                likelihood = log(error_p)
            return likelihood

        likelihoods = [compute_single_likelihood(d, updateflag=True) for d in data]
        self.likelihood = sum(likelihoods) / self.likelihood_temperature
        if updateflag:
            self.update_posterior()
        return self.likelihood

    def compile_function(self):
        self.value_set = None
        return LOTHypothesis.compile_function(self)

    def __call__(self, *args, **kwargs):
        if self.value_set is None:
            value_set = LOTHypothesis.__call__(self)
            # Restrict our concept to being within our domain; also handle 'None' call values
            if isinstance(value_set, set):
                value_set = [x for x in value_set if x <= self.domain]
            else:
                value_set = []
            self.value_set = value_set

        return self.value_set




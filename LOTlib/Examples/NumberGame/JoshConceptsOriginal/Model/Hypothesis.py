
from math import log
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis


class JoshConceptsHypothesis(LOTHypothesis):
    """Class for original josh concepts number game."""
    def __init__(self, grammar, alpha=0.9, domain=100, **kwargs):
        LOTHypothesis.__init__(self, grammar, args=[], **kwargs)
        self.alpha = alpha
        self.domain = domain

    def compute_likelihood(self, data, **kwargs):
        """Likelihood of specified data being produced by this hypothesis.

        Probability is alpha if datum in set, (1-alpha) otherwise).

        """
        s = self()      # set of numbers corresponding to this hypothesis
        error_p = (1.-self.alpha) / self.domain

        def compute_single_likelihood(datum):
            if s is not None and datum in s:
                return log(self.alpha/len(s) + error_p)
            else:
                return log(error_p)

        likelihoods = [compute_single_likelihood(d) for d in data]
        self.likelihood = sum(likelihoods) / self.likelihood_temperature
        self.update_posterior()
        return self.likelihood


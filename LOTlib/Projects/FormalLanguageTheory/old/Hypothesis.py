from LOTlib.Miscellaneous import logsumexp, attrmem
from Levenshtein import distance
from math import log
from LOTlib.Hypotheses.Likelihoods.StochasticLikelihood import StochasticLikelihood
from LOTlib.Hypotheses.RecursiveLOTHypothesis import RecursiveLOTHypothesis
from LOTlib.Eval import RecursionDepthException
from LOTlib.Hypotheses.FactorizedDataHypothesis import FactorizedLambdaHypothesis, FactorizedDataHypothesis
from LOTlib.Hypotheses.FactorizedDataHypothesis import InnerHypothesis

from LOTlib.Miscellaneous import q, Infinity, attrmem


class MyHypothesis(StochasticLikelihood, FactorizedLambdaHypothesis):
    """
    A particular instantiation of FactorizedDataHypothesis, with a likelihood function based on
    levenshtein distance (with small noise rate -- corresponding to -100*distance)
    """

    def __init__(self, **kwargs):
        FactorizedLambdaHypothesis.__init__(self, recurse_bound=5, maxnodes=125, **kwargs)

    def make_hypothesis(self, **kwargs):
        return InnerHypothesis(**kwargs)

    @attrmem('likelihood')
    def compute_likelihood(self, data, shortcut=None):
        # We'll just be overwriting this
        assert len(data) == 1
        return self.compute_single_likelihood(data[0])

    def compute_single_likelihood(self, datum):
        """
            Compute the likelihood with a Levenshtein noise model
        """

        assert isinstance(datum.output, dict), "Data supplied must be a dict (function outputs to counts)"

        llcounts = self.make_ll_counts(datum.input, nsamples=1024)

        lo = sum(llcounts.values())

        ll = 0.0 # We are going to compute a pseudo-likelihood, counting close strings as being close
        for k in datum.output.keys():
            ll += datum.output[k] * (log(llcounts.get(k, 1.0e-12)) - log(lo))
            # ll += datum.output[k] * (log(llcounts.get(k,1.0e-12)) - log(lo)) # Just have a smoothing parameter -- may work well if we use the averaged data
            # ll += datum.output[k] * logsumexp([ log(llcounts[r])-log(lo) - 1000.0 * distance(r, k) for r in llcounts.keys() ])
        return ll


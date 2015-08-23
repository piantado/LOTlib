from LOTlib.Miscellaneous import logsumexp
from Levenshtein import distance
from math import log
from LOTlib.Hypotheses.Likelihoods.StochasticFunctionLikelihood import StochasticFunctionLikelihood
from LOTlib.Hypotheses.RecursiveLOTHypothesis import RecursiveLOTHypothesis
from LOTlib.Evaluation.EvaluationException import RecursionDepthException
from LOTlib.Hypotheses.FactorizedDataHypothesis import FactorizedDataHypothesis
from LOTlib.Hypotheses.FactorizedDataHypothesis import InnerHypothesis
from Grammar import get_Grammar


class FormalLanguageHypothesis(StochasticFunctionLikelihood, RecursiveLOTHypothesis):
    def __init__(self, grammar=None, **kwargs):
        RecursiveLOTHypothesis.__init__(self, grammar, args=[], recurse_bound=20, maxnodes=100, **kwargs)

    def __call__(self, *args):
        try:
            return RecursiveLOTHypothesis.__call__(self, *args)
        except RecursionDepthException:  # catch recursion and too big
            return None


class AnBnCnHypothesis(StochasticFunctionLikelihood, FactorizedDataHypothesis):
    """
    A particular instantiation of FactorizedDataHypothesis, with a likelihood function based on
    levenshtein distance (with small noise rate -- corresponding to -100*distance)
    """

    def __init__(self, **kwargs):
        FactorizedDataHypothesis.__init__(self, recurse_bound=25, maxnodes=125, **kwargs)

    def make_hypothesis(self, **kwargs):
        return InnerHypothesis(**kwargs)

    def compute_single_likelihood(self, datum):
        """
            Compute the likelihood with a Levenshtein noise model
        """

        assert isinstance(datum.output, dict), "Data supplied must be a dict (function outputs to counts)"

        llcounts = self.make_ll_counts(datum.input)

        lo = sum(llcounts.values())

        ll = 0.0 # We are going to compute a pseudo-likelihood, counting close strings as being close
        for k in datum.output.keys():
            ll += datum.output[k] * logsumexp([ log(llcounts[r])-log(lo) - 100.0 * distance(r, k) for r in llcounts.keys() ])
        return ll


class SimpleEnglishHypothesis(AnBnCnHypothesis):

    def __init__(self, **kwargs):
        AnBnCnHypothesis.__init__(self, N=6, **kwargs)


def make_hypothesis(s, **kwargs):
    t = kwargs.pop('terminals') if 'terminals' in kwargs else None
    return AnBnCnHypothesis(grammar=get_Grammar(s, terminals=t), **kwargs)

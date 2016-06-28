from StochasticFunctionLikelihood import StochasticFunctionLikelihood
from LOTlib.Miscellaneous import logsumexp
from LOTlib.Hypotheses.Hypothesis import Hypothesis
from Levenshtein import distance
from math import log

class LevenshteinLikelihood(Hypothesis):
    """
    A (pseudo)likelihood function that is e^(-string edit distance)
    """

    def compute_single_likelihood(self, datum, distance_factor=1.0):
        return -distance_factor*distance(datum.output, self(*datum.input))


class StochasticLevenshteinLikelihood(StochasticFunctionLikelihood):
    """
    A levenshtein distance metric on likelihoods, where the output of a program is corrupted by
    levenshtein noise. This allows for a smoother space of hypotheses over strings.

    Since compute_likelihood passes **kwargs to compute_single_likelihood, we can pass distance_factor
    to compute_likelihood to get it here.
    """

    def compute_single_likelihood(self, datum, llcounts, distance_factor=100.0):
        assert isinstance(datum.output, dict), "Data supplied must be a dict (function outputs to counts)"

        lo = sum(llcounts.values()) # normalizing constant

        # We are going to compute a pseudo-likelihood, counting close strings as being close
        return sum([datum.output[k]*logsumexp([log(llcounts[r])-log(lo) - distance_factor*distance(r, k) for r in llcounts.keys()]) for k in datum.output.keys()])

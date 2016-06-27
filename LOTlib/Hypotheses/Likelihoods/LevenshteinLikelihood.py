from StochasticFunctionLikelihood import StochasticFunctionLikelihood
from LOTlib.Miscellaneous import logsumexp
from Levenshtein import distance
from math import log

class LevenshteinLikelihood(StochasticFunctionLikelihood):
    """
    A levenshtein distance metric on likelihoods, where the output of a program is corrupted by
    levenshtein noise. This allows for a smoother space of hypotheses over strings.

    Since compute_likelihood passes **kwargs to compute_single_likelihood, we can pass distance_factor
    to compute_likelihood to get it here.
    """

    def compute_single_likelihood(self, datum, distance_factor=100.0):
        assert isinstance(datum.output, dict), "Data supplied must be a dict (function outputs to counts)"

        llcounts = self.make_ll_counts(datum.input)

        lo = sum(llcounts.values()) # normalizing constant

        # We are going to compute a pseudo-likelihood, counting close strings as being close
        return sum([datum.output[k]*logsumexp([log(llcounts[r])-log(lo) - distance_factor*distance(r, k) for r in llcounts.keys()]) for k in datum.output.keys()])

"""

    This likelihood is for stochastic functions. To compute the likelihood, we must simulate forwards a bunch of times.
    (Previously, this was a hypothesis type, SimpleGenerativeHypothesis)

    NOTE: A very subtle error can occur if exceptions (like TooBigException) are caught in __call__, then ll_counts may never get set.
"""

from LOTlib.Hypotheses.Hypothesis import Hypothesis
from LOTlib.Miscellaneous import attrmem, nicelog, Infinity
from collections import Counter

class StochasticLikelihood(Hypothesis):

    @attrmem('ll_counts')
    def make_ll_counts(self, input, nsamples=512):
        """
            Run this model forward nsamples times (defaultly self.nsamples),
            returning a dictionary of how often each outcome occurred
        """

        llcounts = Counter()

        for _ in xrange(nsamples):
            llcounts[self(*input)] += 1

        return llcounts

    # def set_value(self, *args, **kwargs):
    #     ret = super(StochasticLikelihood, self).set_value(self, *args, **kwargs)
    #     ret.ll_counts = None # We must recompute these
    #     return ret

    def compute_single_likelihood(self, datum, llcounts, sm=0.1):
        """
                sm smoothing counts are added to existing bins of counts
        """

        assert isinstance(datum.output, dict), "Data supplied to SimpleGenerativeHypothesis must be a dict of function outputs to counts"

        z = sum(llcounts.values())
        return sum([ datum.output[k] * (nicelog(llcounts[k] + sm)-nicelog(sum(z) + sm*len(datum.output.keys())) ) for k in datum.output.keys() ])


    @attrmem('likelihood')
    def compute_likelihood(self, data, shortcut=-Infinity, nsamples=512, **kwargs):
        # For each input, if we don't see its input (via llcounts), recompute it through simulation

        seen = {} # hash of data inputs to llcounts for that input
        ll = 0.0
        for datum in data:

            if datum.input not in seen:
                seen[datum.input] = self.make_ll_counts(datum.input, nsamples=nsamples)

            ll += self.compute_single_likelihood(datum, seen[datum.input] ) / self.likelihood_temperature

            if ll < shortcut:
                return -Infinity

        return ll
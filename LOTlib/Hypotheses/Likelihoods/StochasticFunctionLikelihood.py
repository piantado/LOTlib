"""

    This likelihood is for stochastic functions. To compute the likelihood, we must simulate forwards a bunch of times.
    (Previously, this was a hypothesis type, SimpleGenerativeHypothesis)
"""

from LOTlib.Miscellaneous import attrmem, nicelog
from collections import Counter

class StochasticFunctionLikelihood(object):

    @attrmem('ll_counts')
    def make_ll_counts(self, input, nsamples=512):
        """
            Run this model forward nsamples times (defaultly self.nsamples),
            returning a dictionary of how often each outcome occurred
        """

        if nsamples is None:
            nsamples = self.nsamples

        llcounts = Counter()
        for i in xrange(nsamples):
            llcounts[self(*input)] += 1

        return llcounts


    def compute_single_likelihood(self, datum, llcounts=None, nsamples=512, sm=0.1):
        """
                sm smoothing counts are added to existing bins of counts (just to prevent badness)
                This can take an optiona llcounts in order to allow us to cache this externally
        """
        #print self
        assert isinstance(datum.output, dict), "Data supplied to SimpleGenerativeHypothesis must be a dict (function outputs to counts)"

        if llcounts is None: # compute if not passed in
            llcounts = self.make_ll_counts(datum.input, nsamples=nsamples)

        return sum([ datum.output[k] * (nicelog(llcounts[k] + sm)-nicelog(nsamples + sm*len(datum.output.keys())) ) for k in datum.output.keys() ])


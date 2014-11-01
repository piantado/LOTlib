from LOTHypothesis import LOTHypothesis
from collections import defaultdict
from math import log
from LOTlib.Miscellaneous import Infinity


def nicelog(x):
    if x > 0.:
        return log(x)
    else:
        return -Infinity

class SimpleGenerativeHypothesis(LOTHypothesis):
    """
            Here, each data point is a mapping from input to a dict of counts of observed outputs,
            where if you have K total outputs (sum(output.values()), that counts as K datapoints

            Each function eval results in a string, and the likelihood is the probability of generating the
            observed data. The function eval results in a thunk

            NOTE: FOR NOW, Insert/Delete moves are taken off because of some weirdness with the lambda thunks
    """
    def __init__(self, grammar, nsamples=100, sm=0.001, **kwargs):
        """ kwargs should include ll_sd """

        self.nsamples = nsamples
        self.sm = sm

        LOTHypothesis.__init__(self, grammar,  **kwargs) # this is simple-generative since args=[] (a thunk)


    def make_ll_counts(self,*input):
        """
            Run this model forward nsamples times, returning a dictionary of how often each outcome occurred
        """
        llcounts = defaultdict(int)
        for i in xrange(self.nsamples):
            llcounts[self(*input)] += 1

        self.llcounts = llcounts # we also store this for easy access in the future

        return llcounts


    def compute_single_likelihood(self, datum, llcounts=None):
        """
                sm smoothing counts are added to existing bins of counts (just to prevent badness)
                This can take an optiona llcounts in order to allow us to cache this externally
        """
        #print self
        assert isinstance(datum.output, dict), "Data supplied to SimpleGenerativeHypothesis must be a dict (function outputs to counts)"

        if llcounts is None: # compute if not passed in
            llcounts = self.make_ll_counts(*datum.input)

        return sum([ datum.output[k] * (nicelog(llcounts[k] + self.sm)-nicelog(self.nsamples + self.sm*len(datum.output.keys())) ) for k in datum.output.keys() ])


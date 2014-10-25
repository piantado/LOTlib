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

    def compute_single_likelihood(self, datum):
        assert False, "Should not call this!"

    def compute_single_likelihood(self, datum):
        """
                sm smoothing counts are added to existing bins of counts (just to prevent badness)
        """
        #print self
        assert isinstance(datum.output, dict), "Data supplied to SimpleGenerativeHypothesis must be a dict (function outputs to counts)"

        self.llcounts = defaultdict(int)
        for i in xrange(self.nsamples):
            self.llcounts[ self(*datum.input) ] += 1

        return sum([ datum.output[k] * (nicelog(self.llcounts[k] + self.sm)-nicelog(self.nsamples + self.sm*len(datum.output.keys())) ) for k in datum.output.keys() ]) / self.likelihood_temperature


from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from math import log

from Grammar import grammar

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# A class for scheme hypotheses that just computes the input/output pairs with the appropriate probability

class SchemeFunction(LOTHypothesis):

    # Prior, proposals, __init__ are all inherited from LOTHypothesis
    def __init__(self, ALPHA=0.9, **kwargs):
        LOTHypothesis.__init__(self, grammar, **kwargs)
        self.ALPHA = ALPHA

    def compute_single_likelihood(self, datum):
        """
            Wrap in string for comparisons here. Also, this is a weird pseudo-likelihood (an outlier process)
            since when we are wrong, it should generate the observed data with some probability that's not going
            to be 1-ALPHA
        """
        if str(self(datum.input)) == str(datum.output):
            return log(self.ALPHA)
        else:
            return log(1.0-self.ALPHA)

def make_hypothesis():
    return SchemeFunction()

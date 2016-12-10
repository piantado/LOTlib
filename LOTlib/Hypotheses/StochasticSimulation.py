
from LOTlib.Hypotheses.Hypothesis import Hypothesis
from collections import Counter

class StochasticSimulation(Hypothesis):

    def __call__(self, nsamples=1024, *input):
        """ Overwrite call with a dictionary of outputs """

        output = Counter()
        for _ in xrange(nsamples):
            v = super(self)(*input)
            output[v] += 1
        return output

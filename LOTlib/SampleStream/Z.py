
from LOTlib.Miscellaneous import Infinity, logplusexp
from math import isnan

from SampleStream import SampleStream

class Z(SampleStream):
    """
    This will take a SampleStream and store the logsumexp of all posterior_scores
    """

    def __init__(self, key='posterior_score'):
        SampleStream.__init__(self)

        self.key = key
        self.Z = -Infinity


    def process(self, x):
        v = getattr(x, self.key)
        if not isnan(v):
            self.Z = logplusexp(self.Z, v)

        return x

    def __str__(self):
        return '# Z = %s' % self.Z

    def __exit__(self, t, value, traceback):

        print self

        return SampleStream.__exit__(self, t, value, traceback)



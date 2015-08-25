
from LOTlib.Miscellaneous import Infinity, logplusexp
from math import isnan

from SampleStream import SampleStream

class Z(SampleStream):
    """
    This will take a SampleStream and store the logsumexp of all posterior_scores
    """

    def __init__(self, generator=None, key='posterior_score'):
        self.__dict__.update(locals())

        self.Z = -Infinity

        SampleStream.__init__(self, generator)


    def process_(self, x):
        v = getattr(x, self.key)
        if not isnan(v):
            self.Z = logplusexp(self.Z, v)

    def __str__(self):
        return '# Z = %s' % self.Z




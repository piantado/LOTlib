
from LOTlib.Miscellaneous import Infinity, logplusexp
from math import isnan

from SampleFilter import SampleFilter

class Z(SampleFilter):
    """
    This will take a generator and store the logsumexp of all posterior_scores, perhaps counting only unique ones

    z = Z()
    for x in lot_iter(z(sampler)):
        print x

    print z
    """

    def __init__(self, key='posterior_score', unique=False):
        self.Z = -Infinity
        self.key = key
        self.unique = unique

        self.set = set()

    def add(self, x):
        if (not self.unique) or x not in self.set:
            v = getattr(x, self.key)
            if not isnan(v):
                self.Z = logplusexp(self.Z, v)

        if self.unique:
            self.set.add(x)

    def __str__(self):
        return '# Z = %s' % self.Z





from LOTlib.Miscellaneous import Infinity, logplusexp
from math import isnan

from SampleStream import SampleStream

class Z(SampleStream):
    """
    This will take a generator and store the logsumexp of all posterior_scores, perhaps counting only unique ones

    z = Z()
    for x in break_ctrlc(z(sampler)):
        print x

    print z
    """

    def __init__(self, generator=None, key='posterior_score'):
        self.__dict__.update(locals())

        self.Z = -Infinity

        SampleStream.__init__(self, generator)


    def add(self, x):
        if (not self.unique) or x not in self.set:
            v = getattr(x, self.key)
            if not isnan(v):
                self.Z = logplusexp(self.Z, v)

        if self.unique:
            self.set.add(x)

    def __str__(self):
        return '# Z = %s' % self.Z




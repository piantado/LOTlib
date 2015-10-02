from LOTlib.Miscellaneous import logsumexp, qq
from LOTlib.FiniteBestSet import FiniteBestSet
import pickle
from SampleStream import SampleStream

class Top(SampleStream):
    """
    This wraps "TopN", which is an object to store the best of some samples. This defaultly stores
    them into "file" (since otherwise the top are hard to access)
    """

    def __init__(self, N=1000, key='posterior_score', path=None):
        self.__dict__.update(locals())
        self.fbs = FiniteBestSet(N=N, key=key)

        self.actions = []  # to keep it as a functioning SampleStream

    def process_(self, x):
        self.fbs.add(x)
        return x

    def display(self):
        for h in self.fbs.get_all():
            print h.posterior_score, h.prior, h.likelihood, qq(h)

    def __exit__(self, t, value, traceback):
        """ I defaultly call all of my children's exits """
        for a in self.actions:
            a.__exit__(t, value, traceback)

        # And then save if I have a file
        if self.path is not None:
            f = open(self.path, 'w')
            pickle.dump(self.fbs.get_all(), f)
            f.close()
        return False



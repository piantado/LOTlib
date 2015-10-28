from LOTlib.Miscellaneous import logsumexp, qq
from LOTlib.TopN import TopN
import pickle
from SampleStream import SampleStream

class Top(SampleStream):
    """
    This stores the top samples and then *only* on exit does it allow them to pass through. (It can't before
    exit since it won't know what the samples are!)
    """

    def __init__(self, N=1000, key='posterior_score', sorted=True):
        """

        :param N: How many samples to store.
        :param key:  The key we sort by
        :param sorted: When we output, do we output sorted? (slightly slower)
        :return:
        """
        self.__dict__.update(locals())
        SampleStream.__init__(self)
        self.top = TopN(N=N, key=key)

    def process(self, x):
        """ Overwrite process so all outputs are NOT sent to children.
        """
        self.top.add(x)
        return None # Do no pass through

    def __exit__(self, t, value, traceback):

        ## Here, only on exit do I give my data (the tops) to my outputs
        for v in self.top.get_all(sorted=sorted):
            # Cannot just call self.process_and_push since self.process always returns None
            if v is not None:
                for a in self.outputs:
                    a.process_and_push(v)

        return SampleStream.__exit__(self, t,value,traceback)




class SampleStream(object):
    """
    SampleStream is the basic class that others in this directory inherit from. It allows constructs
    where we process and pass on samples, as in

        pt = PosteriorTrace(plot_every=1000, window=False)
        for h in break_ctrlc(pt(MHSampler(make_hypothesis, data)))
            print h

    or

        for h in break_ctrlc(PosteriorTrace(MHSampler(make_hypothesis, data)))
            print h

    """

    def __init__(self, generator=None):
        self.__dict__.update(locals())

    def update(self, generator):
        for x in generator:
            self.add(x)
            yield x

    def __iter__(self):
        for x in self.update(self.generator):
            yield x

    def add(self, x):
        """
        Add a sample x to myself. Must be overwritten by subclasses
        """
        raise NotImplementedError

    def __call__(self, generator):
        for x in self.update(generator):
            yield x

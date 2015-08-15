import collections

class SampleStream(object):
    """
    SampleStream is the basic class that others in this directory inherit from. It allows constructs
    where we process and pass on samples. (See example below)
    """

    def __init__(self, generator=None):
        self.actions = [] # what do I do to things?
        self.generator = generator

    def update(self, generator):
        """ When given a generator, I will __enter__ and __exit__ (and on all of my actions)"""
        with self:
            for x in generator:
                v = self.process(x)
                if v is not None: # none means skip
                    yield v

    def process(self, x):

        v = self.process_(x)
        for a in self.actions:
            if v is None: break
            v = a.process(v)
        return v

    def process_(self, x):
        """ The main processing step that must be overridden. If it returns x, the sample continues in the stream;
            if it return None, the sample is deleted """
        return x

    def __rshift__(self, other):
        """
        Implements the shift operator. So

        A >> B

        calls A.__rshift__(B).

        We have to return a new SampleStream object or else we can't chain >>
        """

        self.actions.append(other)

        return self

    def __iter__(self):
        assert self.generator is not None, "*** Cannot iter without a generator"
        for x in self.update(self.generator):
            yield x

    def __call__(self, x):
        if isinstance(x, collections.Iterable):
            self.update(x)
        else:
            self.process(x)

    def __enter__(self):
        for a in self.actions:
            a.__enter__()

    def __exit__(self, t, value, traceback):
        """ I defaultly call all of my children's exits """
        for a in self.actions:
            a.__exit__(t, value, traceback)
        return False


if __name__ == "__main__":

    from LOTlib import break_ctrlc
    from LOTlib.Examples.Number.Model import make_hypothesis, make_data
    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
    from __init__ import *

    sampler = break_ctrlc(MHSampler(make_hypothesis(), make_data()))

    tn = TopN(N=10)

    for h in SampleStream(sampler) >> PosteriorTrace(plot_every=100) >> tn >> Save('hypotheses.pkl') \
            >> Tee( Unique() >> PrintH(), Skip(30) >> Print(prefix="#\t")):
        pass

    tn.display()






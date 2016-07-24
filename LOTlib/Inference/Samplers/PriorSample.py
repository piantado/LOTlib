"""
        Inference - sample from the prior (as a baseline comparison)
"""
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Miscellaneous import Infinity, self_update

class PriorSampler(object):
    """
    Just sample from the prior.
    (Only implemented for LOTHypothesis)
    """

    def __init__(self, h0, data, steps=Infinity):
        self_update(self, locals())
        assert isinstance(h0, LOTHypothesis) # only implemented for LOTHypothesis
        self.samples_yielded = 0

    def __iter__(self):
        return self

    def next(self):
        if self.samples_yielded == self.steps:
            raise StopIteration
        else:
            self.samples_yielded += 1
            h = type(self.h0)(self.h0.grammar, start=self.h0.value.returntype)
            h.compute_posterior(self.data)

            return h


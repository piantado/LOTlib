"""
    TODO: add fitting of alpha, baserate, memory decay, etc.
"""
from math import log
from Grammar import grammar
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis

class SFLHypothesis(LOTHypothesis):
    def __init__(self, value=None, alpha=0.99, baserate=0.5):
        LOTHypothesis.__init__(self, grammar, value=value, display='lambda S, x: %s', alpha=alpha, baserate=baserate)

    def evaluate_on_set(self, s):
        return [self(s,x) for x in s]

    def compute_single_likelihood(self, datum):

        v = self.evaluate_on_set(datum.input)

        assert len(v) == len(datum.output)

        ll = 0.0
        for vi, ri in zip(v, datum.output):
            ll += log( self.alpha * (vi==ri) + (1.-self.alpha) * (self.baserate if ri else 1.-self.baserate))

        return ll

def make_hypothesis(**kwargs):
    return SFLHypothesis(**kwargs)
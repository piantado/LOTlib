
from math import log

from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood

from Grammar import grammar

class MagnetismHypothesis(BinaryLikelihood, LOTHypothesis):
    def __init__(self, **kwargs ):
        LOTHypothesis.__init__(self, grammar, args=['x', 'y'], **kwargs)

def make_hypothesis():
    return MagnetismHypothesis()


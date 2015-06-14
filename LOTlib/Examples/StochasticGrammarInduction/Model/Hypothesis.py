
from LOTlib.Hypotheses.Likelihoods.StochasticFunctionLikelihood import StochasticFunctionLikelihood
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from Grammar import grammar

class MyHypothesis(StochasticFunctionLikelihood, LOTHypothesis):
    def __init__(self, grammar=None, **kwargs):
        LOTHypothesis.__init__(self, grammar, args=[''], **kwargs)

def make_hypothesis():
    return MyHypothesis(grammar)
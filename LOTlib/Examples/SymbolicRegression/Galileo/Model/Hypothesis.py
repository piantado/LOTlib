from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Likelihoods.GaussianLikelihood import GaussianLikelihood
from LOTlib.Examples.SymbolicRegression.Grammar import grammar

class MyHypothesis(GaussianLikelihood, LOTHypothesis):
    pass

def make_hypothesis(**kwargs):
    return MyHypothesis(grammar=grammar, **kwargs)

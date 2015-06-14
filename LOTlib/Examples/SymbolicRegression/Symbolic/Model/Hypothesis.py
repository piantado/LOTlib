from LOTlib.Examples.SymbolicRegression.Grammar import grammar
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Likelihoods.GaussianLikelihood import GaussianLikelihood

class MyHypothesis(GaussianLikelihood, LOTHypothesis):
    pass

def make_hypothesis(**kwargs):
    return MyHypothesis(grammar=grammar, **kwargs)
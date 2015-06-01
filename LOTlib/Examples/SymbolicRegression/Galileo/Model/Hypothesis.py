from LOTlib.Hypotheses.GaussianLOTHypothesis import GaussianLOTHypothesis
from Grammar import grammar

def make_hypothesis(**kwargs):
    return GaussianLOTHypothesis(grammar, **kwargs)

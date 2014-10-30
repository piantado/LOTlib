from LOTlib.Hypotheses.GaussianLOTHypothesis import GaussianLOTHypothesis
from Grammar import grammar

def make_h0(**kwargs):
    return GaussianLOTHypothesis(grammar, **kwargs)

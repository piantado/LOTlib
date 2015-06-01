
from LOTlib.Hypotheses.GaussianLOTHypothesis import GaussianLOTHypothesis
from LOTlib.Examples.SymbolicRegression.Grammar import grammar

def make_hypothesis():
    return GaussianLOTHypothesis(grammar)

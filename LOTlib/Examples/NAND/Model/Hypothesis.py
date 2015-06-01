from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from Grammar import grammar

def make_hypothesis():
    return LOTHypothesis(grammar, start='START', args=['x'])
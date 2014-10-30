
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from Grammar import grammar

def make_h0(**kwargs):
    return LOTHypothesis(grammar, args=['x', 'y'], ALPHA=0.999, **kwargs) # alpha here trades off with the amount of data. Currently assuming no noise, but that's not necessary


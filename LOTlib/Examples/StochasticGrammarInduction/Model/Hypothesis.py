
from LOTlib.Hypotheses.SimpleGenerativeHypothesis import SimpleGenerativeHypothesis
from Grammar import grammar

def make_hypothesis():
    return SimpleGenerativeHypothesis(grammar, args=[''] )
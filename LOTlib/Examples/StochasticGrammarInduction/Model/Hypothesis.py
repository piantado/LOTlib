
from LOTlib.Hypotheses.SimpleGenerativeHypothesis import SimpleGenerativeHypothesis
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from Grammar import grammar

class MyHypothesis(SimpleGenerativeHypothesis, LOTHypothesis):
    def __init__(self, grammar, **kwargs):
        SimpleGenerativeHypothesis.__init__(self)
        LOTHypothesis.__init__(self, grammar, args=[''], **kwargs)

def make_hypothesis():
    return MyHypothesis(grammar)
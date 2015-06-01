from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis

from Grammar import grammar

def make_hypothesis():
    # Create an initial hypothesis -- defaultly generated at random from the grammar
    return LOTHypothesis(grammar, args=['S'])

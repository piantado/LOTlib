from LOTlib.Inference.MetropolisHastings import mh_sample
from Data import data
from Grammar import grammar
from Specification import SchemeFunction

"""
    A LOTlib example for bound variables.

    This does inference over cons, cdr, car expressions.
    NOTE: it does not work very well for complex functions since it is hard to sample a close function -- not much of a gradient to climb on cons, cdr, car
"""

STEPS = 1000000
ALPHA = 0.9

# And run
h0 = SchemeFunction(grammar, ALPHA=ALPHA)
for x in mh_sample(h0, data, STEPS):

    print x.posterior_score, x
    for di in data:
        print "\t", di.input, "->", x(*di.input), " ; should be ", di.output

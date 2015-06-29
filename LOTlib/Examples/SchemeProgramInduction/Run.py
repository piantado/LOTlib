
"""
A LOTlib example for bound variables.

This does inference over cons, cdr, car expressions.

Note:
    it does not work very well for complex functions since it is hard to sample a close function -- not
    much of a gradient to climb on cons, cdr, car

"""
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib import break_ctrlc
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis

from Model import *
from Model.Grammar import grammar
from Model.Data import make_data

#=============================================================================================================

STEPS = 1000000
ALPHA = 0.9

def run():

    h0 = make_hypothesis()
    data = make_data()

    for x in break_ctrlc(MHSampler(h0, data, STEPS)):

        print x.posterior_score, x
        for di in data:
            print "\t", di.input, "->", x(*di.input), " ; should be ", di.output


#=============================================================================================================

example_input = [
    [],
    [[]],
    [[], []],
    [[[]]]
]


def scheme_generate():
    """ This generates random scheme code with cons, cdr, and car, and evaluates it on some simple list
    structures.

    No inference here -- just random sampling from a grammar.

    """
    ## Generate some and print out unique ones
    seen = set()
    for i in break_ctrlc(xrange(10000)):
        x = grammar.generate('START')

        if x not in seen:
            seen.add(x)

            # make the function node version
            f = LOTHypothesis(grammar, value=x, args=['x'])

            print x.log_probability(), x
            for ei in example_input:
                print "\t", ei, " -> ", f(ei)

if __name__ == "__main__":
    run()

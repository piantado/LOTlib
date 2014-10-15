
"""
    This generates random scheme code with cons, cdr, and car, and evaluates it on some simple list structures.
    No inference here -- just random sampling from a grammar.
"""

from LOTlib.Hypotheses import LOTHypothesis
from Grammar import grammar

example_input = [   [], [[]], [ [], [] ], [[[]]]  ]

## Generate some and print out unique ones
seen = set()
for i in xrange(10000):
    x = grammar.generate('START')

    if x not in seen:
        seen.add(x)

        # make the function node version
        f = LOTHypothesis(grammar, value=x, args=['x'])

        print x.log_probability(), x
        for ei in example_input:
            print "\t", ei, " -> ", f(ei)

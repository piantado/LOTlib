"""
An example to show running number
"""
from LOTHypothesisState import *
from LOTlib import lot_iter

from LOTlib.Examples.Number2015 import grammar, generate_data, get_knower_pattern, make_h0
data = generate_data(400)

for _ in lot_iter(xrange(10000)):
    print grammar.nonterminals()
    for x in grammar.nonterminals():
        if len(grammar.get_rules(x)) > 0: # could be zero if this nt is only added in bv
            print x, grammar.generate(x=x)


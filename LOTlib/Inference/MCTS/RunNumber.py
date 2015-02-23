"""
An example to show running number
"""
from LOTHypothesisState import *
from LOTlib import lot_iter

from LOTlib.Examples.Number2015 import grammar, generate_data, get_knower_pattern, make_h0
data = generate_data(400)

s = LOTHypothesisState.make(make_h0, data, grammar, C=1500.0, V=10.0)

for x in lot_iter(s):
    print x.posterior_score, x.prior, x.likelihood, get_knower_pattern(x), x

print "<><><><><><><><><><><><><><><><><><><><>"
s.show_graph(maxdepth=3)
print "<><><><><><><><><><><><><><><><><><><><>"
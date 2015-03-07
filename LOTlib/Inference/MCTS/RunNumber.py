"""
An example to show running number
"""
from LOTHypothesisState import *
from LOTlib import break_ctrlc

from LOTlib.Examples.Number import grammar, generate_data, get_knower_pattern, make_h0
data = generate_data(100)

s = LOTHypothesisState.make(make_h0, data, grammar, C=100.0)

for i, x in enumerate(break_ctrlc(s)):
    print i, x.posterior_score, x.prior, x.likelihood, get_knower_pattern(x), x

print "<><><><><><><><><><><><><><><><><><><><>"
s.show_graph(maxdepth=3)
print "<><><><><><><><><><><><><><><><><><><><>"
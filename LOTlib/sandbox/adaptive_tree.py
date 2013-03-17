
"""

	Try out generating from a PCFG and then adapting the probabilities

"""

from LOTlib.Grammars import Grammar_SimpleBoolean

G = Grammar_SimpleBoolean
G.add_rule('BOOL', 'x', [], 10.0)


for x in 
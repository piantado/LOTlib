
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q

SHAPES = ['square', 'triangle', 'rectangle']
COLORS = ['blue', 'red', 'green']

# ------------------------------------------------------------------
# Set up the grammar
# Here, we create our own instead of using DefaultGrammars.Nand because
# we don't want a BOOL/PREDICATE distinction
# ------------------------------------------------------------------
FEATURE_WEIGHT = 2. # Probability of expanding to a terminal

grammar = Grammar()

grammar.add_rule('START', '', ['BOOL'], 1.0)

grammar.add_rule('BOOL', 'nand_', ['BOOL', 'BOOL'], 1.0/3.)
grammar.add_rule('BOOL', 'nand_', ['True', 'BOOL'], 1.0/3.)
grammar.add_rule('BOOL', 'nand_', ['False', 'BOOL'], 1.0/3.)

# And finally, add the primitives
for s in SHAPES:
    grammar.add_rule('BOOL', 'is_shape_', ['x', q(s)], FEATURE_WEIGHT)

for c in COLORS:
    grammar.add_rule('BOOL', 'is_color_', ['x', q(c)], FEATURE_WEIGHT)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TERMINAL_WEIGHT = 5.0

from LOTlib.Grammar import Grammar

grammar = Grammar()
grammar.add_rule('START', '', ['EXPR'], 1.0)

grammar.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], 1.0)
grammar.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], 1.0)
grammar.add_rule('EXPR', 'divide_', ['EXPR', 'EXPR'], 1.0)
grammar.add_rule('EXPR', 'subtract_', ['EXPR', 'EXPR'], 1.0)

grammar.add_rule('EXPR', 'exp_', ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'log_', ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'pow_', ['EXPR', 'EXPR'], 1.0) # including this gives lots of overflow

grammar.add_rule('EXPR', 'sin_', ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'cos_', ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'tan_', ['EXPR'], 1.0)

grammar.add_rule('EXPR', 'x', None, TERMINAL_WEIGHT) # these terminals should have None for their function type; the literals

grammar.add_rule('EXPR', '1.0', None, TERMINAL_WEIGHT)

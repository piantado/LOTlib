__author__ = 'eric'

from LOTlib.Grammar import Grammar

# ------------------------------------------------------------------------------------------------------------
# This grammar has 20 rules    |Expressions| * |Constants|

simple_grammar = Grammar()
simple_grammar.add_rule('START', '', ['SET'], 1.)

# Mapping expressions over sets of numbers
simple_grammar.add_rule('SET', 'mapset_', ['FUNC', 'RANGE'], 1.)
simple_grammar.add_rule('RANGE', 'range_set_', ['1', '100'], 1.)
simple_grammar.add_rule('FUNC', 'lambda', ['EXPR'], 1., bv_type='X', bv_p=1.)

# Expressions
simple_grammar.add_rule('EXPR', 'times_', ['X', '1'], 1.)
simple_grammar.add_rule('EXPR', 'times_', ['X', '2'], 1.)
simple_grammar.add_rule('EXPR', 'times_', ['X', '3'], 1.)
simple_grammar.add_rule('EXPR', 'times_', ['X', '4'], 1.)
simple_grammar.add_rule('EXPR', 'times_', ['X', '5'], 1.)
simple_grammar.add_rule('EXPR', 'times_', ['X', '6'], 1.)
simple_grammar.add_rule('EXPR', 'times_', ['X', '7'], 1.)
simple_grammar.add_rule('EXPR', 'times_', ['X', '8'], 1.)
simple_grammar.add_rule('EXPR', 'times_', ['X', '9'], 1.)
simple_grammar.add_rule('EXPR', 'times_', ['X', '10'], 1.)

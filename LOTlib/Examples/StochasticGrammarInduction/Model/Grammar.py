import LOTlib.Miscellaneous
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
## Let's add a primitive to our evaluation.
## The alternative is to use the decorator @primitive from LOTlib.Evaluation.Eval

from LOTlib.Evaluation.Eval import register_primitive

register_primitive(LOTlib.Miscellaneous.flatten2str)

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

TERMINAL_WEIGHT = 2.

grammar = Grammar()

# flattern2str lives at the top, and it takes a cons, cdr, car structure and projects it to a string
grammar.add_rule('START', 'flatten2str', ['EXPR'], 1.0)

grammar.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 1.)
grammar.add_rule('BOOL', 'or_', ['BOOL', 'BOOL'], 1.)
grammar.add_rule('BOOL', 'not_', ['BOOL'], 1.)

grammar.add_rule('EXPR', 'if_', ['BOOL', 'EXPR', 'EXPR'], 1.)
grammar.add_rule('BOOL', 'equal_', ['EXPR', 'EXPR'], 1.)

grammar.add_rule('BOOL', 'flip_', [''], TERMINAL_WEIGHT)

# List-building operators
grammar.add_rule('EXPR', 'cons_', ['EXPR', 'EXPR'], 1.)
grammar.add_rule('EXPR', 'cdr_', ['EXPR'], 1.)
grammar.add_rule('EXPR', 'car_', ['EXPR'], 1.)

grammar.add_rule('EXPR', '[]', None, TERMINAL_WEIGHT)
grammar.add_rule('EXPR', q('D'), None, TERMINAL_WEIGHT)
grammar.add_rule('EXPR', q('A'), None, TERMINAL_WEIGHT)
grammar.add_rule('EXPR', q('N'), None, TERMINAL_WEIGHT)
grammar.add_rule('EXPR', q('V'), None, TERMINAL_WEIGHT)
grammar.add_rule('EXPR', q('who'), None, TERMINAL_WEIGHT)


## Allow lambda abstraction
grammar.add_rule('EXPR', 'apply_', ['LAMBDAARG', 'LAMBDATHUNK'], 1)
grammar.add_rule('LAMBDAARG',   'lambda', ['EXPR'], 1., bv_type='EXPR', bv_args=[] )
grammar.add_rule('LAMBDATHUNK', 'lambda', ['EXPR'], 1., bv_type=None, bv_args=None ) # A thunk


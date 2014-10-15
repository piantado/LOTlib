import LOTlib.Miscellaneous
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
## Here is an example of using define_for_evaluator
from LOTlib.Evaluation.Eval import register_primitive # this creates new defines
# And this calls them:
register_primitive(LOTlib.Miscellaneous.flatten2str)

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
TERMINAL_WEIGHT = 5.

grammar = Grammar()
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


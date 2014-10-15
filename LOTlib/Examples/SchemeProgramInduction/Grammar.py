from LOTlib.Grammar import Grammar

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# A simple grammar for scheme, including lambda
grammar = Grammar()

# A very simple version of lambda calculus
grammar.add_rule('START', '', ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'apply_', ['FUNC', 'EXPR'], 1.0)
grammar.add_rule('EXPR', 'x', None, 5.0)
grammar.add_rule('FUNC', 'lambda', ['EXPR'], 1.0, bv_type='EXPR', bv_args=None)

grammar.add_rule('EXPR', 'cons_', ['EXPR', 'EXPR'], 1.0)
grammar.add_rule('EXPR', 'cdr_',  ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'car_',  ['EXPR'], 1.0)

grammar.add_rule('EXPR', '[]',  None, 1.0)


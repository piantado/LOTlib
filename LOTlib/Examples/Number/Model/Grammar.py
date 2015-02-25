from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q

# ============================================================================================================
#  Define a PCFG

# The priors here are somewhat hierarchical by type in generation, tuned to be a little more efficient
# (but the actual RR prior does not care about these probabilities)

grammar = Grammar(start='START')

grammar.add_rule('START', '', ['WORD'], 1.0)

grammar.add_rule('BOOL', 'and_',    ['BOOL', 'BOOL'], 1./3.)
grammar.add_rule('BOOL', 'or_',     ['BOOL', 'BOOL'], 1./3.)
grammar.add_rule('BOOL', 'not_',    ['BOOL'], 1./3.)

grammar.add_rule('BOOL', 'True',    None, 1.0/2.)
grammar.add_rule('BOOL', 'False',   None, 1.0/2.)

# note that this can take basically any types for return values
grammar.add_rule('WORD', 'if_',    ['BOOL', 'WORD', 'WORD'], 0.5)
grammar.add_rule('WORD', 'if_',    ['BOOL', 'WORD', q('undef')], 0.5)
# grammar.add_rule('WORD', 'ifU_',    ['BOOL', 'WORD'], 0.5)  # if returning undef if condition not met

grammar.add_rule('BOOL', 'cardinality1_',    ['SET'], 1.0)
grammar.add_rule('BOOL', 'cardinality2_',    ['SET'], 1.0)
grammar.add_rule('BOOL', 'cardinality3_',    ['SET'], 1.0)

grammar.add_rule('BOOL', 'equal_',    ['WORD', 'WORD'], 1.0)

grammar.add_rule('SET', 'union_',     ['SET', 'SET'], 1./3.)
grammar.add_rule('SET', 'intersection_',     ['SET', 'SET'], 1./3.)
grammar.add_rule('SET', 'setdifference_',     ['SET', 'SET'], 1./3.)
grammar.add_rule('SET', 'select_',     ['SET'], 1.0)

grammar.add_rule('SET', 'x',     None, 4.0)

grammar.add_rule('WORD', 'recurse_',        ['SET'], 1.0)

grammar.add_rule('WORD', 'next_', ['WORD'], 1.0)
grammar.add_rule('WORD', 'prev_', ['WORD'], 1.0)

# grammar.add_rule('WORD', 'undef', None, 1.0)

# These are quoted (q) since they are strings when evaled
grammar.add_rule('WORD', q('one_'), None, 0.10)
grammar.add_rule('WORD', q('two_'), None, 0.10)
grammar.add_rule('WORD', q('three_'), None, 0.10)
grammar.add_rule('WORD', q('four_'), None, 0.10)
grammar.add_rule('WORD', q('five_'), None, 0.10)
grammar.add_rule('WORD', q('six_'), None, 0.10)
grammar.add_rule('WORD', q('seven_'), None, 0.10)
grammar.add_rule('WORD', q('eight_'), None, 0.10)
grammar.add_rule('WORD', q('nine_'), None, 0.10)
grammar.add_rule('WORD', q('ten_'), None, 0.10)

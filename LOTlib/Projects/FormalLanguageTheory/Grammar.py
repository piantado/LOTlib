"""
This version uses Flip.py

"""

from LOTlib.Grammar import Grammar

base_grammar = Grammar()
base_grammar.add_rule('START', '', ['LIST'], 1.0)

base_grammar.add_rule('LIST', '(%s if %s else %s)', ['LIST', 'BOOL', 'LIST'], 1.)

base_grammar.add_rule('LIST', 'strcons_', ['LIST', 'LIST'], 3.) # upweighted to help in search/proposals
base_grammar.add_rule('LIST', 'strcdr_', ['LIST'], 1.)
base_grammar.add_rule('LIST', 'strcar_', ['LIST'], 1.)

base_grammar.add_rule('LIST', '', ['ATOM'], 3.0) # Terminals
base_grammar.add_rule('LIST', 'x', None, 3.0) # the argument

# base_grammar.add_rule('LIST', '', ['ATOMSEQ'], 3.0) # Terminals
# base_grammar.add_rule('ATOMSEQ', '%s+%s', ['ATOMSEQ', 'ATOM'], 1.0) # Terminals
# base_grammar.add_rule('ATOMSEQ', '%s', ['ATOM'],            2.0) # Terminals


base_grammar.add_rule('BOOL', 'empty_', ['LIST'], 1.)
base_grammar.add_rule('BOOL', 'C.flip(p=%s)', ['PROB'], 1.) # flip within a context

for i in xrange(5,10):
    base_grammar.add_rule('PROB', '0.%s' % i, None, 1.)

# can call memoized and non-memoized versions of myself
base_grammar.add_rule('LIST', 'lex_(C, %s, %s)',   ['SELFF', 'LIST'], 1.0) # can call other words
base_grammar.add_rule('SELFF', '0', None, 1.0)  #needed to tha tif lex is called on the first word, the grammar still works. It has low prob so its not generated once the other lex are added


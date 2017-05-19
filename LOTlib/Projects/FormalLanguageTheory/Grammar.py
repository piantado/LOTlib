"""
This version uses Flip.py

"""

from LOTlib.Grammar import Grammar

base_grammar = Grammar()
base_grammar.add_rule('START', '', ['START2'], 1.0)
base_grammar.add_rule('START2', '', ['LIST'], 1.0) # just to make isnertions/deletions at the top easier

base_grammar.add_rule('LIST', '(%s if %s else %s)', ['LIST', 'BOOL', 'LIST'],1.)

base_grammar.add_rule('LIST', 'strcons_', ['LIST', 'LIST'], 1.) # upweighted (3?) to help in search/proposals
base_grammar.add_rule('LIST', 'strcdr_', ['LIST'], 1.)
base_grammar.add_rule('LIST', 'strcar_', ['LIST'], 1.)

base_grammar.add_rule('LIST', '', ['ATOM'], 3.0) # Terminals
base_grammar.add_rule('LIST', 'x', None, 3.0) # the argument

# base_grammar.add_rule('LIST', '', ['ATOMSEQ'], 3.0) # Terminals
# base_grammar.add_rule('ATOMSEQ', '%s+%s', ['ATOMSEQ', 'ATOM'], 1.0) # Terminals
# base_grammar.add_rule('ATOMSEQ', '%s', ['ATOM'],            2.0) # Terminals

# If we want to allow sampling from a set of "words"
# base_grammar.add_rule('LIST',     'C.uniform_sample([%s])', ['WORDLIST'], 1.) # Uniform sample of a word list
# base_grammar.add_rule('WORDLIST', '%s', ['WORD'], 1.) # flip within a context
# base_grammar.add_rule('WORDLIST', '%s, %s', ['WORD', 'WORDLIST'], 1.) # flip within a context
# base_grammar.add_rule('WORD', '%s', ['ATOM'], 1.) # flip within a context
# base_grammar.add_rule('WORD', '%s+%s', ['ATOM', 'WORD'], 1.) # flip within a context


base_grammar.add_rule('BOOL', 'empty_', ['LIST'], 1.)
base_grammar.add_rule('BOOL', 'C.flip(p=%s)', ['PROB'], 1.) # flip within a context

for i in xrange(5,10):
    base_grammar.add_rule('PROB', '0.%s' % i, None, 1.)

# can call memoized and non-memoized versions of myself
base_grammar.add_rule('LIST', 'lex_(C, %s, %s)',   ['SELFF', 'LIST'], 1.0) # can call other words
base_grammar.add_rule('SELFF', '0', None, 1.0)  #needed to tha tif lex is called on the first word, the grammar still works. It has low prob so its not generated once the other lex are added


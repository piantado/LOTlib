"""
Here is a version where the string operations manipulate dictionaries mapping strings to probabilities.
This prevents us from having to simulate everything. NOTE: It requires RecursionDepthException to
be handled as generating empty string

TODO: Add BOOL -- and or not, equality, etc.

"""

from LOTlib.Grammar import Grammar

base_grammar = Grammar()
base_grammar.add_rule('START', '', ['LIST'], 1.0)

base_grammar.add_rule('LIST', '(%s if %s else %s)', ['LIST', 'BOOL', 'LIST'], 1.)

base_grammar.add_rule('LIST', 'strcons_', ['LIST', 'LIST'], 1./3.)
base_grammar.add_rule('LIST', 'strcdr_', ['LIST'], 1./3.)
base_grammar.add_rule('LIST', 'strcar_', ['LIST'], 1./3.)

base_grammar.add_rule('LIST', '', ['ATOM'], 1.0)
base_grammar.add_rule('LIST', 'x', None, 5.0) # the argument

base_grammar.add_rule('ATOM', "\'\'", None, 1.0)

base_grammar.add_rule('BOOL', 'empty_', ['LIST'], 1.)
base_grammar.add_rule('BOOL', 'C.flip(p=%s)', ['PROB'], 1.) # flip within a context

for i in xrange(5,10):
    base_grammar.add_rule('PROB', '0.%s' % i, None, 1.)

# can call memoized and non-memoized versions of myself
base_grammar.add_rule('LIST', 'lex_(C, %s, %s)',   ['SELFF', 'LIST'], 1.0) # can call other words, but I have to say whether to memoize or not
base_grammar.add_rule('SELFF', '0', None, 1.0)  #needed to tha tif lex is called on the first word, the grammar still works. It has low prob so its not generated wonce the other lex are added


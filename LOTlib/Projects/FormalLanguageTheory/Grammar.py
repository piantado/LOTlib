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

base_grammar.add_rule('LIST', 'strcons_', ['LIST', 'LIST'], 3.)

# temporarily removed these because they make everything much more complex, unnecessarily so?
# But they are needed for anbncn, right? Perhaps we can just enable recursion on emptry string or strcdr(x)?
# so that we simplify the search process a lot?
base_grammar.add_rule('LIST', 'strcdr_', ['LIST'], 1.)
base_grammar.add_rule('LIST', 'strcar_', ['LIST'], 1.)


from LOTlib.Eval import primitive

# @primitive
# def insert_(l,s,i):
#     return l[:i]+s+l[i:]
#
# base_grammar.add_rule('LIST', 'insert_', ['LIST', 'LIST', 'POSITION'], 1.)
# for n in xrange(1,5):
#     base_grammar.add_rule('POSITION', str(n), None, 1.0)

base_grammar.add_rule('LIST', '', ['ATOM'], 3.0) # Terminals
base_grammar.add_rule('LIST', 'x', None, 3.0) # the argument

# base_grammar.add_rule('ATOM', "\'\'", None, 1.0)

base_grammar.add_rule('BOOL', 'empty_', ['LIST'], 1.)
base_grammar.add_rule('BOOL', 'C.flip(p=%s)', ['PROB'], 1.) # flip within a context

for i in xrange(5,10):
    base_grammar.add_rule('PROB', '0.%s' % i, None, 1.)

# can call memoized and non-memoized versions of myself
base_grammar.add_rule('LIST', 'lex_(C, %s, %s)',   ['SELFF', 'LIST'], 1.0) # can call other words
base_grammar.add_rule('SELFF', '0', None, 1.0)  #needed to tha tif lex is called on the first word, the grammar still works. It has low prob so its not generated once the other lex are added


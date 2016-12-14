"""
Here is a version where the string operations manipulate dictionaries mapping strings to probabilities.
This prevents us from having to simulate everything. NOTE: It requires RecursionDepthException to
be handled as generating empty string
"""

from LOTlib.Grammar import Grammar

base_grammar = Grammar()
base_grammar.add_rule('START', '', ['LIST'], 1.0)

base_grammar.add_rule('LIST', 'if_d', ['BOOL', 'LIST', 'LIST'], 1.)

base_grammar.add_rule('LIST', 'cons_d', ['ATOM', 'LIST'], 1./6.)
base_grammar.add_rule('LIST', 'cons_d', ['LIST', 'LIST'], 1./6.)
base_grammar.add_rule('LIST', 'cdr_d', ['LIST'], 1./3.)
base_grammar.add_rule('LIST', 'car_d', ['LIST'], 1./3.)

base_grammar.add_rule('LIST', '', ['ATOM'], 3.0)
base_grammar.add_rule('LIST', '{\'\':0.0}', None, 1.0)
# base_grammar.add_rule('LIST', 'recurse_', [], 1.) # This is added by factorizedDataHypothesis

base_grammar.add_rule('BOOL', 'empty_d', ['LIST'], 1.)
base_grammar.add_rule('BOOL', 'flip_d(p=%s)', ['PROB'], 1.)

for i in xrange(1,10):
    base_grammar.add_rule('PROB', '0.%s' % i, None, 1.)

# can call memoized and non-memoized versions of myself
base_grammar.add_rule('LIST', 'recurse_(%s, True)',  ['SELFF'], 1.0) # can call myself
base_grammar.add_rule('LIST', 'recurse_(%s, False)', ['SELFF'], 1.0) # can call myself


if __name__ == "__main__":
    from Model import InnerHypothesis

    for _ in xrange(1000):
        h = InnerHypothesis(grammar=base_grammar)
        print h
        print h(h)

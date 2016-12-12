# Yuan's version:
from LOTlib.Grammar import Grammar

base_grammar = Grammar()
base_grammar.add_rule('START', 'flatten2str', ['LIST', 'sep=\"\"'], 1.0)

base_grammar.add_rule('LIST', 'if_', ['BOOL', 'LIST', 'LIST'], 1.)

base_grammar.add_rule('LIST', 'cons_', ['ATOM', 'LIST'], 1./6.)
base_grammar.add_rule('LIST', 'cons_', ['LIST', 'LIST'], 1./6.)
base_grammar.add_rule('LIST', 'cdr_', ['LIST'], 1./3.)
base_grammar.add_rule('LIST', 'car_', ['LIST'], 1./3.)

base_grammar.add_rule('LIST', '', ['ATOM'], 3.0)
base_grammar.add_rule('LIST', '\'\'', None, 1.0)
# base_grammar.add_rule('LIST', 'recurse_', [], 1.) # This is added by factorizedDataHypothesis

base_grammar.add_rule('BOOL', 'empty_', ['LIST'], 1.)
base_grammar.add_rule('BOOL', 'flip_(p=%s)', ['PROB'], 1.)

for i in xrange(1,10):
    base_grammar.add_rule('PROB', '0.%s' % i, None, 1.)

base_grammar.add_rule('LIST', 'recurse_(%s)', ['SELFF'], 1.0) # can call myself


if __name__ == "__main__":

    for _ in xrange(1000):
        print base_grammar.generate()

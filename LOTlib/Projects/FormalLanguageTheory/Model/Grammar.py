# Yuan's version:
from LOTlib.Grammar import Grammar

base_grammar = Grammar()
base_grammar.add_rule('START', 'flatten2str', ['LIST', 'sep=\"\"'], 1.0)
base_grammar.add_rule('LIST', 'if_', ['BOOL', 'LIST', 'LIST'], 1.)
base_grammar.add_rule('LIST', 'cons_', ['ATOM', 'LIST'], 1.)
base_grammar.add_rule('LIST', 'cons_', ['LIST', 'LIST'], 1.)
# base_grammar.add_rule('LIST', 'cdr_', ['LIST'], 1.)
# base_grammar.add_rule('LIST', 'car_', ['LIST'], 1.)

base_grammar.add_rule('LIST', 'optional_',  ['LIST', 'LIST'], 1.)
base_grammar.add_rule('LIST', 'geometric_', ['LIST', 'LIST'], 1.)

base_grammar.add_rule('LIST', '', ['ATOM'], 5.)
# base_grammar.add_rule('LIST', '\'\'', None, 2)
base_grammar.add_rule('LIST', 'recurse_', [], 1.) # This is added by factorizedDataHypothesis

# base_grammar.add_rule('BOOL', 'empty_', ['LIST'], 1.)
base_grammar.add_rule('BOOL', 'flip_', [''], 1.)


from random import random
from LOTlib.Eval import primitive
from LOTlib.Primitives import *

@primitive
def optional_(x, y):
    if random() < 0.5:
        return cons_(x,y)
    else:
        return y

@primitive
def geometric_(x,y):
    # geometric number of xes followed by y
    if random() < 0.5:
        return y
    else:
        return cons_(x, geometric_(x,y))



# from LOTlib.Grammar import Grammar
#
# base_grammar = Grammar()
# base_grammar.add_rule('START', 'flatten2str', ['LIST', 'sep=\"\"'], 1.0)
# base_grammar.add_rule('LIST', 'if_', ['BOOL', 'LIST', 'LIST'], 1.)
# base_grammar.add_rule('LIST', 'cons_', ['ATOM', 'LIST'], 1.)
# base_grammar.add_rule('LIST', 'cons_', ['LIST', 'LIST'], 1.)
# base_grammar.add_rule('LIST', 'cdr_', ['LIST'], 1.)
# base_grammar.add_rule('LIST', 'car_', ['LIST'], 1.)
#
# base_grammar.add_rule('LIST', '', ['ATOM'], 5.)
# base_grammar.add_rule('LIST', '\'\'', None, 2)
# base_grammar.add_rule('LIST', 'recurse_', [], 1.)
#
# base_grammar.add_rule('BOOL', 'empty_', ['LIST'], 1.)
# base_grammar.add_rule('BOOL', 'flip_', [''], 1.)


from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q

base_grammar = Grammar()
base_grammar.add_rule('START', 'flatten2str', ['LIST', 'sep=\"\"'], 1.0)
base_grammar.add_rule('LIST', 'if_', ['BOOL', 'LIST', 'LIST'], 1.)
base_grammar.add_rule('LIST', 'cons_', ['ATOM', 'LIST'], 1.)
base_grammar.add_rule('LIST', 'cons_', ['LIST', 'LIST'], 1.)
base_grammar.add_rule('LIST', 'cdr_', ['LIST'], 1.)
base_grammar.add_rule('LIST', 'car_', ['LIST'], 1.)
base_grammar.add_rule('LIST', '\'\'', None, 2)
# base_grammar.add_rule('LIST', 'recurse_', [], 1.)

base_grammar.add_rule('BOOL', 'empty_', ['LIST'], 1.)
base_grammar.add_rule('BOOL', 'flip_', [''], 1.)

from copy import deepcopy

a_grammar = deepcopy(base_grammar)
for x in 'a':
    a_grammar.add_rule('ATOM', q(x), None, 2)

eng_grammar = deepcopy(base_grammar)
for x in 'davtnih':
    eng_grammar.add_rule('ATOM', q(x), None, 2)


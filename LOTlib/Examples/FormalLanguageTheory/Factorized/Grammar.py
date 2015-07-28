import LOTlib

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Evaluation.Eval import register_primitive
register_primitive(LOTlib.Miscellaneous.flatten2str)

from LOTlib.Miscellaneous import q
from LOTlib.Grammar import Grammar

grammar = Grammar()

# flattern2str lives at the top, and it takes a cons, cdr, car structure and projects it to a string
grammar.add_rule('START', 'flatten2str', ['LIST', 'sep=\"\"'], 1.0)

grammar.add_rule('LIST', 'if_', ['BOOL', 'LIST', 'LIST'], 0.09)
# grammar.add_rule('ATOM', 'if_', ['BOOL', 'ATOM', 'ATOM'], 1.)
grammar.add_rule('BOOL', 'empty_', ['LIST'], 0.56)

grammar.add_rule('BOOL', 'flip_', [''], 0.43)

# List-building operators
grammar.add_rule('LIST', 'cons_', ['ATOM', 'LIST'], 0.203)
grammar.add_rule('LIST', 'cdr_', ['LIST'], 0.15)
grammar.add_rule('LIST', 'car_', ['LIST'], 0.15)

# In the most recent version, both of these are variably set in AnBnCnHypothesis
# grammar.add_rule('LIST', 'recurse_', ['LIST'], 1.)
# grammar.add_rule('LIST', 'lst', None, TERMINAL_WEIGHT) # the argument

grammar.add_rule('LIST', '\'\'', None, 0.23)

grammar.add_rule('ATOM', q('a'), None, .33)
grammar.add_rule('ATOM', q('b'), None, .33)
grammar.add_rule('ATOM', q('c'), None, .33)

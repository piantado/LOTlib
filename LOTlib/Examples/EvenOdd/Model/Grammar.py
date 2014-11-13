from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q

WORDS = ['even', 'odd']

grammar = Grammar()
grammar.add_rule('START', '', ['BOOL'], 1.)

grammar.add_rule('BOOL', 'equals_', ['NUMBER', 'NUMBER'], 1.)
grammar.add_rule('BOOL', 'not_', ['BOOL'], 1.)

grammar.add_rule('BOOL', 'and_sc_', ['BOOL', 'BOOL'], 1.)
grammar.add_rule('BOOL', 'or_sc_',  ['BOOL', 'BOOL'], 1.)  # use the short_circuit form

grammar.add_rule('NUMBER', 'x', None, 1.)
grammar.add_rule('NUMBER', '1', None, 1.)
grammar.add_rule('NUMBER', '0', None, 1.)
grammar.add_rule('NUMBER', 'plus_', ['NUMBER', 'NUMBER'], 1.)
grammar.add_rule('NUMBER', 'minus_', ['NUMBER', 'NUMBER'], 1.)

for w in WORDS:
    grammar.add_rule('BOOL', 'lexicon', [q(w), 'NUMBER'], 1.)
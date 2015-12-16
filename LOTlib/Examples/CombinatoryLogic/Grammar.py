"""
Just create some combinators and reduce them.

"""
from LOTlib.Grammar import Grammar

grammar = Grammar()

grammar.add_rule('START', 'cons_', ['START', 'START'], 2.0)

grammar.add_rule('START', 'I', None, 1.0)
grammar.add_rule('START', 'S', None, 1.0)
grammar.add_rule('START', 'K', None, 1.0)


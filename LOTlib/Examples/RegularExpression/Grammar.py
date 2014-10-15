from LOTlib.Grammar import Grammar

##########################################################
# Define a grammar
grammar = Grammar()

grammar.add_rule('START', '', ['EXPR'], 1.0)

grammar.add_rule('EXPR', 'star_', ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'question_', ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'plus_', ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'or_', ['EXPR', 'EXPR'], 1.0)
grammar.add_rule('EXPR', 'str_append_', ['TERMINAL', 'EXPR'], 5.0)
grammar.add_rule('EXPR', 'terminal_', ['TERMINAL'], 5.0)

for v in 'abc.':
    grammar.add_rule('TERMINAL', v, None, 1.0)


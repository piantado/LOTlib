"""
Set up the grammar.

Note: This was updated on Dec 3 2012, after the language submission. We now include AND/OR/NOT, and S,
and removed nonempty.

"""
from LOTlib.Grammar import Grammar

grammar = Grammar()


grammar.add_rule('START', 'presup_', ['BOOL', 'BOOL'], 1.0)

grammar.add_rule('START', 'presup_', ['True', 'BOOL'], 1.0)
grammar.add_rule('START', 'presup_', ['False', 'BOOL'], 1.0)

grammar.add_rule('START', 'presup_', ['False', 'False'], 1.0)
grammar.add_rule('START', 'presup_', ['True', 'True'], 1.0)

grammar.add_rule('BOOL', 'not_', ['BOOL'], 1.0)
grammar.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'or_', ['BOOL', 'BOOL'], 1.0)
#grammar.add_rule('BOOL', 'nonempty_', ['SET'], 1.0) # don't need this if we do logical operations

grammar.add_rule('BOOL', 'empty_', ['SET'], 1.0)
grammar.add_rule('BOOL', 'subset_', ['SET', 'SET'], 1.0)
grammar.add_rule('BOOL', 'exhaustive_', ['SET', 'context.S'], 1.0)
grammar.add_rule('BOOL', 'cardinality1_', ['SET'], 1.0) # if cardinalities are included, don't include these!
grammar.add_rule('BOOL', 'cardinality2_', ['SET'], 1.0)
grammar.add_rule('BOOL', 'cardinality3_', ['SET'], 1.0)

grammar.add_rule('SET', 'union_', ['SET', 'SET'], 1.0)
grammar.add_rule('SET', 'intersection_', ['SET', 'SET'], 1.0)
grammar.add_rule('SET', 'setdifference_', ['SET', 'SET'], 1.0)

# These will just be attributes of the current context
grammar.add_rule('SET', 'context.A', None, 10.0)
grammar.add_rule('SET', 'context.B', None, 10.0)
grammar.add_rule('SET', 'context.S', None, 10.0) ## Must include this or else we can't get complement

# Cardinality operations
grammar.add_rule('BOOL', 'cardinalityeq_', ['SET', 'SET'], 1.0)
grammar.add_rule('BOOL', 'cardinalitygt_', ['SET', 'SET'], 1.0)
#grammar.add_rule('BOOL', 'cardinalityeq_', ['SET', 'CARD'], 1.0)
#grammar.add_rule('BOOL', 'cardinalitygt_', ['SET', 'CARD'], 1.0)
#grammar.add_rule('BOOL', 'cardinalitygt_', ['CARD', 'SET'], 1.0)
###grammar.add_rule('CARD', 'cardinality_', ['SET'], 1.0)

##grammar.add_rule('CARD',  '0', None, 1.0)
#grammar.add_rule('CARD',  '1', None, 1.0)
#grammar.add_rule('CARD',  '2', None, 1.0)
#grammar.add_rule('CARD',  '3', None, 1.0)
#grammar.add_rule('CARD',  '4', None, 1.0)
#grammar.add_rule('CARD',  '5', None, 1.0)
#grammar.add_rule('CARD',  '6', None, 1.0)
#grammar.add_rule('CARD',  '7', None, 1.0)
#grammar.add_rule('CARD',  '8', None, 1.0)



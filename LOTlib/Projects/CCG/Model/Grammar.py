
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import qq

# What are the objects we may use?
OBJECTS              = ['JOHN', 'MARY', 'SUSAN', 'BILL']
SEMANTIC_1PREDICATES = ['SMILED', 'LAUGHED', 'MAN', 'WOMAN']
SEMANTIC_2PREDICATES = ['SAW', 'LOVED']

## Define the grammar
grammar = Grammar()

grammar.add_rule('START', '', ['FUNCTION'], 2.0)
grammar.add_rule('START', '', ['BOOL'], 1.0)
grammar.add_rule('START', '', ['OBJECT'], 1.0)

for m in SEMANTIC_1PREDICATES:
    grammar.add_rule('BOOL', 'C.relation_', [ qq(m), 'OBJECT'], 1.0)

for m in SEMANTIC_2PREDICATES:
    grammar.add_rule('BOOL', 'C.relation_', [ qq(m), 'OBJECT', 'OBJECT'], 1.0)

for o in OBJECTS:
    grammar.add_rule('OBJECT', qq(o), None, 1.0)

grammar.add_rule('BOOL', 'exists_', ['FUNCTION.O2B', 'C.objects'], 1.00) # can quantify over objects->bool functions
grammar.add_rule('BOOL', 'forall_', ['FUNCTION.O2B', 'C.objects'], 1.00)
grammar.add_rule('FUNCTION.O2B', 'lambda', ['BOOL'], 1.0, bv_type='OBJECT')

grammar.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'or_', ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'not_', ['BOOL'], 1.0)

# And for outermost functions
grammar.add_rule('FUNCTION', 'lambda', ['START1'], 1.0, bv_type='OBJECT')
grammar.add_rule('FUNCTION', 'lambda', ['START2'], 1.0, bv_type='BOOL', bv_args=['OBJECT'])
grammar.add_rule('FUNCTION', 'lambda', ['START3'], 1.0, bv_type='BOOL', bv_args=['OBJECT', 'OBJECT'])

grammar.add_rule('START1', '', ['START'], 1.0)
grammar.add_rule('START2', '', ['START'], 1.0)
grammar.add_rule('START3', '', ['START'], 1.0)
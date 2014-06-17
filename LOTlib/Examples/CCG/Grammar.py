
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import qq
from Shared import OBJECTS, SEMANTIC_1PREDICATES, SEMANTIC_2PREDICATES


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
grammar.add_rule('FUNCTION', 'lambda', ['START'], 1.0, bv_type='OBJECT')
grammar.add_rule('FUNCTION', 'lambda', ['START'], 1.0, bv_type='BOOL', bv_args=['OBJECT'])
grammar.add_rule('FUNCTION', 'lambda', ['START'], 1.0, bv_type='BOOL', bv_args=['OBJECT', 'OBJECT'])

from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis

# How we make a hypothesis inside the lexicon
def make_hypothesis(): 
    return LOTHypothesis(grammar, args=['C'])
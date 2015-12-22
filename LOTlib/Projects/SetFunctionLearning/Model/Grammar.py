

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define primitives

from LOTlib.Eval import register_primitive

@register_primitive
def circle_(x):
    return x.shape=="circle"
@register_primitive
def triangle_(x):
    return x.shape=="triangle"
@register_primitive
def rectangle_(x):
    return x.shape=="rectangle"

@register_primitive
def yellow_(x):
    return x.color=="yellow"
@register_primitive
def green_(x):
    return x.color=="green"
@register_primitive
def blue_(x):
    return x.color=="blue"

@register_primitive
def size1_(x):
    return x.size==1
@register_primitive
def size2_(x):
    return x.size==2
@register_primitive
def size3_(x):
    return x.size==3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Add in predicates

from LOTlib.Grammar import Grammar

grammar = Grammar()

grammar.add_rule('START', 'False', None, 1.0)
grammar.add_rule('START', 'True', None, 1.0)
grammar.add_rule('START', '', ['BOOL'], 10.0)

grammar.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'or_', ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'not_', ['BOOL'], 1.0)
grammar.add_rule('BOOL', 'iff_', ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'implies_', ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'xor_', ['BOOL', 'BOOL'], 1.0)

grammar.add_rule('BOOL', '', ['PREDICATE'], 10.0)  # Upweight to make well-formed

for prd in ["circle_", "triangle_", "rectangle_", "yellow_", "green_", "blue_", "size1_", "size2_", "size3_"]:
    grammar.add_rule('PREDICATE', prd, ['OBJECT'], 1.0)

grammar.add_rule('OBJECT', 'x', None, 1.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# And add in quantification

grammar.add_rule('BOOL', 'forall_', ['OBJECT2BOOL', 'SET'], 1.0)
grammar.add_rule('BOOL', 'exists_', ['OBJECT2BOOL', 'SET'], 1.0)

grammar.add_rule('OBJECT2BOOL', 'lambda', ['BOOL'], 1.0, bv_type='OBJECT')

grammar.add_rule('SET', 'S', None, 1.0)
grammar.add_rule('SET', '(set(S)-set([x]))', None, 1.0)













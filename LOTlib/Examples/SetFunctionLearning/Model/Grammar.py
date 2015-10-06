
from LOTlib.DefaultGrammars import SimpleBoolean as grammar
from LOTlib.Evaluation.Eval import register_primitive

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

for prd in ["circle_", "triangle_", "rectangle_", "yellow_", "green_", "blue_", "size1_", "size2_", "size3_"]:

    grammar.add_rule('PREDICATE', prd, ['x'], 1.0)
















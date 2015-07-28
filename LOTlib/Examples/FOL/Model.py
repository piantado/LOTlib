

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Grammar import Grammar
grammar = Grammar()

grammar.add_rule('START', '', ['QUANT'], 1.0)

# Very simple -- one allowed and required quantifier
grammar.add_rule('QUANT', 'exists_', ['FUNCTION', 'SET'], 1.00)
grammar.add_rule('QUANT', 'forall_', ['FUNCTION', 'SET'], 1.00)

# The thing we are a function of
grammar.add_rule('SET', 'S', None, 1.0)

# And allow us to create a new kind of function
grammar.add_rule('FUNCTION', 'lambda', ['BOOL'], 1.0, bv_type='OBJECT')

# Logical connectives
grammar.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'or_', ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'not_', ['BOOL'], 1.0)

# non-terminal arguments get passed as normal python arguments
grammar.add_rule('BOOL', 'is_color_',  ['OBJECT', '\'red\''],   5.00) # --> is_color_(OBJECT, 'red') --> OBJECT.color == 'red'
grammar.add_rule('BOOL', 'is_color_',  ['OBJECT', '\'blue\''],  5.00)
grammar.add_rule('BOOL', 'is_color_',  ['OBJECT', '\'green\''], 5.00)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Hypothesis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood

class MyHypothesis(BinaryLikelihood, LOTHypothesis):
    def __init__(self, grammar=grammar, **kwargs):
        LOTHypothesis.__init__(self, grammar=grammar, args=['S'], **kwargs)

def make_hypothesis():
    return MyHypothesis()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.DataAndObjects import FunctionData, Obj # for nicely managing data

# Make up some data -- here just one set containing {red, red, green} colors that is mapped to True
def make_data():
    return [ FunctionData(input=[ {Obj(color='red'), Obj(color='red'), Obj(color='green')} ], output=True, alpha=0.99) ]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    from LOTlib.Examples.Demo import standard_sample

    standard_sample(make_hypothesis, make_data, save_top=False)








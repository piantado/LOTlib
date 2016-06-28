

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
        LOTHypothesis.__init__(self, grammar=grammar, display='lambda S: %s', **kwargs)

def make_hypothesis(**kwargs):
    return MyHypothesis(**kwargs)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.DataAndObjects import FunctionData, Obj # for nicely managing data

# Make up some data -- here just one set containing {red, red, green} colors that is mapped to True
def make_data(n=1):
    return [ FunctionData(input=[ {Obj(color='red'), Obj(color='red'), Obj(color='green')} ], output=True, alpha=0.99) ]*n

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    from LOTlib.Inference.Samplers.StandardSample import standard_sample

    standard_sample(make_hypothesis, make_data, save_top=False)

    #
    # from LOTlib import break_ctrlc
    # from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
    #
    # h0 = make_hypothesis()
    # data = make_data(10)
    #
    # for h in break_ctrlc(MHSampler(h0, data, skip=100)):
    #     print h.posterior_score, grammar.pack_ascii(h.value), h







SHAPES = ['square', 'triangle', 'rectangle']
COLORS = ['blue', 'red', 'green']

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q

FEATURE_WEIGHT = 2. # Probability of expanding to a terminal

# Set up the grammar
# Here, we create our own instead of using DefaultGrammars.Nand because
# we don't want a BOOL/PREDICATE distinction
grammar = Grammar()

grammar.add_rule('START', '', ['BOOL'], 1.0)

grammar.add_rule('BOOL', 'nand_', ['BOOL', 'BOOL'], 1.0/3.)
grammar.add_rule('BOOL', 'nand_', ['True', 'BOOL'], 1.0/3.)
grammar.add_rule('BOOL', 'nand_', ['False', 'BOOL'], 1.0/3.)

# And finally, add the primitives
for s in SHAPES:
    grammar.add_rule('BOOL', 'is_shape_', ['x', q(s)], FEATURE_WEIGHT)

for c in COLORS:
    grammar.add_rule('BOOL', 'is_color_', ['x', q(c)], FEATURE_WEIGHT)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.DataAndObjects import FunctionData, make_all_objects
from LOTlib.Miscellaneous import sample_one

all_objects = make_all_objects( shape=SHAPES, color=COLORS )

# Generator for data

# Just import some defaults
from LOTlib.Examples.NAND.TargetConcepts import TargetConcepts

def make_data(N=20, f=TargetConcepts[0]):
    data = []
    for _ in xrange(N):
        o = sample_one(all_objects)
        data.append( FunctionData(input=[o], output=f(o), alpha=0.90) )
    return data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Hypothesis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood

class MyHypothesis(BinaryLikelihood, LOTHypothesis):
    def __init__(self, grammar=grammar, **kwargs):
        LOTHypothesis.__init__(self, grammar=grammar, display="lambda x : %s", **kwargs)

def make_hypothesis(**kwargs):
    return MyHypothesis(**kwargs)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    from LOTlib.Inference.Samplers.StandardSample import standard_sample

    standard_sample(make_hypothesis, make_data, save_top=False)
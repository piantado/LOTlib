
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.DataAndObjects import FunctionData, Obj

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# from LOTlib.DefaultGrammars import SimpleBoolean

# DNF defaultly includes the logical connectives so we need to add predicates to it.
# grammar = SimpleBoolean

from LOTlib.Grammar import Grammar

grammar = Grammar()

grammar.add_rule('START', '', ['PREDICATE'], 1.0)

# Predicates for color, shape, size
grammar.add_rule('PREDICATE', 'is_color_(x, "red")', None, 1.0)
grammar.add_rule('PREDICATE', 'is_color_(x, "green")', None, 1.0)
grammar.add_rule('PREDICATE', 'is_color_(x, "blue")', None, 1.0)

grammar.add_rule('PREDICATE', 'is_shape_(x, "circle")', None, 1.0)
grammar.add_rule('PREDICATE', 'is_shape_(x, "square")', None, 1.0)
grammar.add_rule('PREDICATE', 'is_shape_(x, "triangle")', None, 1.0)

grammar.add_rule('PREDICATE', 'is_size_(x, "small")',  None,  1.0)
grammar.add_rule('PREDICATE', 'is_size_(x, "medium")',  None,  1.0)
grammar.add_rule('PREDICATE', 'is_size_(x, "large")',  None,  1.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Hypothesis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Priors.RationalRules import RationaRulesPrior
from LOTlib.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood

class MyHypothesis(RationaRulesPrior,BinaryLikelihood, LOTHypothesis):
    pass


def make_hypothesis(grammar=grammar, **kwargs):
    return MyHypothesis(grammar=grammar, rrAlpha=1.0, **kwargs)



from math import log

from LOTlib.Grammar import Grammar

grammar = Grammar()

# shape, color, size
# DATA = image e.g., RED_TRIANGLE_LARGE
# Hypothesis: function(image) = Bool

grammar.add_rule('START', '', ['PREDICATE'], 1.0)

grammar.add_rule('START', 'True', None, 1.0)
grammar.add_rule('START', 'False', None, 1.0)

# Logical Primitives
grammar.add_rule('PREDICATE', 'and_', ['PREDICATE', 'PREDICATE'], 1.0)
grammar.add_rule('PREDICATE', 'or_', ['PREDICATE', 'PREDICATE'], 1.0)
grammar.add_rule('PREDICATE', 'not_', ['PREDICATE'], 1.0)

# Color Primitives
grammar.add_rule('PREDICATE', 'red_', ['IMG'], 1.0)
grammar.add_rule('PREDICATE', 'green_', ['IMG'], 1.0)

# Shape Primitives
grammar.add_rule('PREDICATE', 'triangle_', ['IMG'], 1.0)
grammar.add_rule('PREDICATE', 'square_', ['IMG'], 1.0)

# Size Primitives
grammar.add_rule('PREDICATE', 'large_', ['IMG'], 1.0)
grammar.add_rule('PREDICATE', 'small_', ['IMG'], 1.0)

from LOTlib.Eval import primitive

@primitive
def red_(img):
    return 'RED' in img.split('_')

@primitive
def green_(img):
    return 'GREEN' in img.split('_')

@primitive
def triangle_(img):
    return 'TRIANGLE' in img.split('_')

@primitive
def square_(img):
    return 'SQUARE' in img.split('_')

@primitive
def large_(img):
    return 'LARGE' in img.split('_')

@primitive
def small_(img):
    return 'SMALL' in img.split('_')

from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Likelihoods.PowerLawDecayed import PowerLawDecayed
from LOTlib.Miscellaneous import attrmem
class MyHypothesis(PowerLawDecayed, LOTHypothesis):
    def __init__(self, **kwargs):
        LOTHypothesis.__init__(self, grammar=grammar, display="lambda IMG: %s", **kwargs)

    def compute_single_likelihood(self, datum, **kwargs):
        ll = log(datum.alpha * (self(*datum.input) == datum.output) + (1.0 - datum.alpha) / 2.0)

        return ll

    @attrmem('posterior_no_prior')
    def compute_posterior_no_prior(self, d, **kwargs):
        return self.compute_likelihood(d, **kwargs)

from LOTlib.DataAndObjects import FunctionData
data = [FunctionData(input = ["GREEN_TRIANGLE_SMALL"], output = 1, alpha = 0.9),
        FunctionData(input = ["RED_SQUARE_LARGE"], output = 0, alpha = 0.9)]

if __name__ == "__main__":
    h0 = MyHypothesis()
    h0.ll_decay = 0.

    from LOTlib.TopN import TopN
    hyps = TopN(N = 1000)

    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
    from LOTlib import break_ctrlc
    mhs = MHSampler(h0, data, 1000000, likelihood_temperature = 1., prior_temperature = 1.)

    for samples_yielded, h in break_ctrlc(enumerate(mhs)):
        h.ll_decay = 0.
        hyps.add(h)

    import pickle
    with open('HypothesisSpace2.pkl', 'w') as f:
        pickle.dump(hyps, f)

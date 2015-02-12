"""
A simple rational rules demo.

This script scatters our imports around to show where each part comes from.

Paper:
    A rational analysis of rule-based concept learning. N. D. Goodman, J. B. Tenenbaum, J. Feldman, and T. L.
    Griffiths (2008). Cognitive Science. 32:1, 108-154. http://www.mit.edu/~ndg/papers/RRfinal3.pdf

"""
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from Model import *


def run_mh():
    """Run the MH; Run the vanilla sampler.

    Without steps, it will run infinitely. This prints out posterior (posterior_score), prior, tree grammar
    probability, likelihood,

    This yields data like below:
        -10.1447997767 -9.93962659915 -12.2377573418 -0.20517317755 'and_(not_(is_shape_(x, 'triangle')),
            not_(is_color_(x, 'blue')))'
        -11.9260879461 -8.77647578935 -12.2377573418 -3.14961215672 'and_(not_(is_shape_(x, 'triangle')),
            not_(is_shape_(x, 'triangle')))'

    """
    # Create an initial hypothesis. Here we use a RationalRulesLOTHypothesis, which
    # is defined in LOTlib.Hypotheses and wraps LOTHypothesis with the rational rules prior
    h0 = RationalRulesLOTHypothesis(grammar=DNF, rrAlpha=1.0)

    for h in lot_iter(MHSampler(h0, data, 10000, skip=100)):
        print h.posterior_score, h.prior, h.value.log_probability(), h.likelihood, q(h)


if __name__ == "__main__":
    run_mh()
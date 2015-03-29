# -*- coding: utf-8 -*-
"""
Symbolic regression that fits parameters using a MAP estimate of the continuous parameters.

We put a simple bayesian prior on these constants, and use it to compute MAPs.

"""
from math import sin
import numpy

from LOTlib import break_ctrlc
from LOTlib.Miscellaneous import qq
from Grammar import grammar
from Hypothesis import MAPSymbolicRegressionHypothesis
from Data import generate_data


STEPS = 500000
SKIP = 0
data_sd = 0.1 # the SD of the data
NDATA = 50
MEMOIZE = 1000 # 0 means don't memoize

## The target function for symbolic regression
target = lambda x: 3.*x + sin(4.3/x)

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# starting hypothesis -- here this generates at random
def run():
    data = generate_data(target, NDATA, data_sd) # generate some data
    h0 = MAPSymbolicRegressionHypothesis(grammar, args=['x']+CONSTANT_NAMES)
    h0.CONSTANT_VALUES = numpy.zeros(NCONSTANTS) ## TODO: Move this to an itializer

    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
    for h in break_ctrlc(MHSampler(h0, data, STEPS, skip=SKIP, trace=False)):
        print h.posterior_score, h.likelihood, h.prior, h.CONSTANT_VALUES, qq(h)


if __name__ == "__main__":
    run()
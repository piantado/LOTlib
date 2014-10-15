# -*- coding: utf-8 -*-

"""

         A demo of "syntax" learning using a SimpleGenerativeHypothesis.

         This searches over probabilistic generating functions, running them forward to estimate
         the likelihood of the data. Very very simple.
"""
from LOTlib.Inference.MetropolisHastings import mh_sample
from LOTlib.Hypotheses.SimpleGenerativeHypothesis import SimpleGenerativeHypothesis
from Data import data
from Grammar import grammar


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
h0 = SimpleGenerativeHypothesis(grammar )

## populate the finite sample by running the sampler for this many steps
for h in mh_sample(h0, data, 100000, skip=100):
    print h.posterior_score, h.prior, h.likelihood, h
    print h.llcounts

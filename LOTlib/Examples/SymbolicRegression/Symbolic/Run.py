# -*- coding: utf-8 -*-
"""
A simple symbolic regression demo

"""
from LOTlib import break_ctrlc
from LOTlib.Hypotheses.GaussianLOTHypothesis import GaussianLOTHypothesis
from LOTlib.Inference.MetropolisHastings import MHSampler
from LOTlib.Miscellaneous import qq
from LOTlib.Examples.SymbolicRegression.Grammar import grammar
from Data import generate_data

CHAINS = 4
STEPS = 50000
SKIP = 0

if __name__ == "__main__":

    print grammar

    # generate some data
    data = generate_data(50) # how many data points?

    # starting hypothesis -- here this generates at random
    h0 = GaussianLOTHypothesis(grammar)

    for h in break_ctrlc(MHSampler(h0, data, STEPS, skip=SKIP)):
        print h.posterior_score, qq(h)
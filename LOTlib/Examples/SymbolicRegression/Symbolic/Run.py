# -*- coding: utf-8 -*-
"""
A simple symbolic regression demo

"""
from LOTlib.Hypotheses.GaussianLOTHypothesis import GaussianLOTHypothesis
from LOTlib.Inference.MetropolisHastings import mh_sample
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.DataAndObjects import qq
from LOTlib.Examples.SymbolicRegression.Grammar import grammar
from Data import generate_data


CHAINS = 4
STEPS = 50000
SKIP = 0

# generate some data
data = generate_data(50) # how many data points?


def run(*args):
    """one run with these parameters"""
    # starting hypothesis -- here this generates at random
    h0 = GaussianLOTHypothesis(grammar)

    # We store the top 100 from each run
    fs = FiniteBestSet(10, max=True, key="posterior_score")
    fs.add(  mh_sample(h0, data, STEPS, skip=SKIP)  )

    return fs


def multirun():
    """Multicore, parallel."""
    finitesample = FiniteBestSet(max=True) # the finite sample of all
    results = map(run, [ [] ] * CHAINS ) # a not parallel
    finitesample.merge(results)

    for r in finitesample.get_all():
        print r.posterior_score, r.prior, r.likelihood, qq(str(r))

if __name__ == "__main__":
    run()

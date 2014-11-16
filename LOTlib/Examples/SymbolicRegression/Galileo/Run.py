# -*- coding: utf-8 -*-
"""
This uses Galileo's data on a falling ball.

See: http://www.amstat.org/publications/jse/v3n1/datasets.dickey.html
See also: Jeffreys, W. H., and Berger, J. O. (1992), "Ockham's Razor and Bayesian Analysis," American
    Scientist, 80, 64-72 (Erratum, p. 116).

"""
from LOTlib.Hypotheses.GaussianLOTHypothesis import GaussianLOTHypothesis
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.Inference.MetropolisHastings import mh_sample
from LOTlib.Miscellaneous import qq
from Data import data
from Grammar import grammar
from Utilities import make_h0


def run(*args):
    """The running function."""
    # starting hypothesis -- here this generates at random
    h0 = GaussianLOTHypothesis(grammar)

    # We store the top 100 from each run
    pq = FiniteBestSet(N=100, max=True, key="posterior_score")
    pq.add(mh_sample(h0, data, STEPS, skip=SKIP))

    return pq


if __name__ == "__main__":

    CHAINS = 10
    STEPS = 10000000
    SKIP = 0

    finitesample = FiniteBestSet(max=True) # the finite sample of all
    results = map(run, [ [None] ] * CHAINS ) # Run on a single core
    finitesample.merge(results)

    ## and display
    for r in finitesample.get_all(decreasing=False, sorted=True):
        print r.posterior_score, r.prior, r.likelihood, qq(str(r))

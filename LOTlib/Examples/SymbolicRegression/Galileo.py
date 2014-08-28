# -*- coding: utf-8 -*-
from LOTlib.Hypotheses.GaussianLOTHypothesis import GaussianLOTHypothesis
from LOTlib.DataAndObjects import FunctionData
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.Inference.MetropolisHastings import mh_sample
from LOTlib.Miscellaneous import qq
from Grammar import grammar

"""
        This uses Galileo's data on a falling ball. See: http://www.amstat.org/publications/jse/v3n1/datasets.dickey.html
        See also, Jeffreys, W. H., and Berger, J. O. (1992), "Ockham's Razor and Bayesian Analysis," American Scientist, 80, 64-72 (Erratum, p. 116).
"""

# NOTE: these must be floats, else we get hung up on powers of ints
data_sd = 50.0
data = [
         FunctionData(input=[1000.], output=1500., ll_sd=data_sd),
         FunctionData(input=[828.], output=1340., ll_sd=data_sd),
         FunctionData(input=[800.], output=1328., ll_sd=data_sd),
         FunctionData(input=[600.], output=1172., ll_sd=data_sd),
         FunctionData(input=[300.], output=800., ll_sd=data_sd), 
         FunctionData(input=[0.], output=0., ll_sd=data_sd) # added 0,0 since it makes physical sense.
        ]

CHAINS = 10
STEPS = 10000000
SKIP = 0
PRIOR_TEMPERATURE=1.0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Grammar import Grammar

grammar = Grammar()
grammar.add_rule('START', '', ['EXPR'], 1.0)

grammar.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], 1.0)
grammar.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], 1.0)
grammar.add_rule('EXPR', 'divide_', ['EXPR', 'EXPR'], 1.0)
grammar.add_rule('EXPR', 'subtract_', ['EXPR', 'EXPR'], 1.0)

grammar.add_rule('EXPR', 'exp_', ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'log_', ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'pow_', ['EXPR', 'EXPR'], 1.0) # including this gives lots of overflow

grammar.add_rule('EXPR', 'sin_', ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'cos_', ['EXPR'], 1.0)
grammar.add_rule('EXPR', 'tan_', ['EXPR'], 1.0)

grammar.add_rule('EXPR', 'x', None, 5.0) # these terminals should have None for their function type; the literals

grammar.add_rule('EXPR', '1.0', None, 5.0)

# # # # # # # # # # # # # # # # # # # # # # # # #
# Standard exports

def make_h0(value=None):
    return  GaussianLOTHypothesis(grammar, value=value)

if __name__ == "__main__":

    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # the running function

    def run(*args):

        # starting hypothesis -- here this generates at random
        h0 = GaussianLOTHypothesis(grammar, prior_temperature=PRIOR_TEMPERATURE)

        # We store the top 100 from each run
        pq = FiniteBestSet(100, max=True, key="posterior_score")
        pq.add( mh_sample(h0, data, STEPS, skip=SKIP)  )

        return pq

    finitesample = FiniteBestSet(max=True) # the finite sample of all
    results = map(run, [ [None] ] * CHAINS ) # Run on a single core
    finitesample.merge(results)

    ## and display
    for r in finitesample.get_all(decreasing=False, sorted=True):
        print r.posterior_score, r.prior, r.likelihood, qq(str(r))

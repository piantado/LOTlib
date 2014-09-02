from LOTlib import lot_iter

from LOTlib.Grammar import Grammar
from LOTlib.Hypotheses.LOTHypothesis import *
from LOTlib.Evaluation.Eval import *
from LOTlib.Miscellaneous import logsumexp, exp
from math import log


#~~~~~~~~~~~~~~~~~~~~~~~~ 1 : Wrapper class for hypothesis ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
class NumberGameExpression(LOTHypothesis):

    def __init__(self, grammar, domain=100, args=['n'], **kwargs):
        LOTHypothesis.__init__(self, grammar, args=args, **kwargs)
        self.domain = domain

    def compute_likelihood(self, data, noise=0.9):
        """ Computes the likelihood of data. """

        # Get domain subset that data would map to using hypothesis function
        subset = map(self, map(float, range(1, self.domain + 1)))
        subset = [item for item in subset if item <= self.domain]

        # The likelihood of each element d in data will depend on whether d is in subset
        s = len(subset)
        self.likelihood = 0
        
        for datum in data:
            if datum in subset:
                self.likelihood += log(noise / s)
            else:   # If datum not in hypo subset OR hypo subset is empty, use noise = (1-alpha)
                self.likelihood += log((1-noise) / self.domain)

        # This is required in all compute_likelihoods
        self.posterior_score = self.prior + self.likelihood
        return self.likelihood


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2 : Create grammar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# - Rules in our grammar map between sets of integers in our domain                     #
# - E.g. "3n+1" maps to { 1, 4, 7, 10, ... }                                            #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
grammar = Grammar()
grammar.add_rule('START', '', ['EXPR'], 1)
grammar.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], 1)
grammar.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], 1)
grammar.add_rule('EXPR', 'pow_', ['EXPR', 'EXPR'], 0.5)

# Converted to digits so python can evaluate them
grammar.add_rule('EXPR', 'n', None, 4)
for i in xrange(1, 10):
    grammar.add_rule('EXPR', str(i), None, 0.10)


#~~~~~~~~~~~~~~~~~~~~~~~~ 3 : Generate hypotheses for data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
num_iters = 100
data = [2, 8, 16]

hypotheses = set()
for _ in lot_iter(xrange(num_iters)):       # You can ctrl+C to exit this loop
    t = NumberGameExpression(grammar)
    t.compute_posterior(data)               # Get prior, likelihood, posterior
    hypotheses.add(t)

# the exp(posterior_score) is proportional to posterior probability, so we can estimate the normalizing constant Z like this:
Z = logsumexp([h.posterior_score for h in hypotheses])

for h in sorted(hypotheses, key=lambda x: x.posterior_score):
    print h.prior, h.likelihood, exp(h.posterior_score - Z), h

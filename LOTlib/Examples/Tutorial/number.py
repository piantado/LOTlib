from LOTlib import lot_iter                             # lets you ctrl-C out of a loop

from LOTlib.Grammar import Grammar
from LOTlib.Hypotheses.LOTHypothesis import *
from LOTlib.Evaluation.Eval import *
from LOTlib.Miscellaneous import logsumexp, exp
from math import log

grammar = Grammar()
grammar.add_rule('START', '', ['EXPR'], 1)

# all rules are set operations over a set of integers
grammar.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], 1)
grammar.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], 1)
grammar.add_rule('EXPR', 'pow_', ['EXPR', 'EXPR'], 0.5)

## These have been converted to digits so that when python evals them, they have the right value.
grammar.add_rule('EXPR', 'n', None, 4)
for i in xrange(1, 10):
    grammar.add_rule('EXPR', str(i), None, 0.10)

class NumberGameExpression(LOTHypothesis):
    """ Hypothesis """

    def __init__(self, grammar, domain=100, args=['n'], **kwargs):
        LOTHypothesis.__init__(self, grammar, args=args, **kwargs)
        self.domain = domain

    def compute_likelihood(self, data):
        """ Computes the likelihood of data. """
        n = len(data)

        # integer arithmetic is really slow for things like pow => convert to float
        subset = map(self, map(float, range(1, self.domain + 1)))
        subset = [item for item in subset if item <= self.domain]

## The below is almost the right idea.
## Remember here that "data" may be a list or set, and you need to compute the likelihood of *every* element
## in that list/set. The likelihood of each element d in data will depend on whether d is in subset and
## How many elements are in subset. How likely are you to choose each d in data from subset?
        s = len(subset)
        if s == 0:
            self.likelihood = -Infinity
        else:
            self.likelihood = log((1. / s) ** n)  ## This is not quite right (mathematically), but compute_liklihood has to set self.likelihood and self.posterior_score

        ## This is required in all compute_likelihoods.
        ####""" It would be nice to simplify so you write it as you did """
        ## If you define compute_single_likelihood (e.g. on a single data point), then you don't need to do this
        self.posterior_score = self.prior + self.likelihood
        return self.likelihood


num_iters = 100
data = [2, 8, 16]

hypotheses = set()
for _ in lot_iter(xrange(num_iters)):
    t = NumberGameExpression(grammar)
    t.compute_posterior(data)               # Call compute_posterior to get the prior, likelihood, and posterior
    hypotheses.add(t)                       # LOTHypotheses hash nicely

# .posterior_score is equal to .prior + .likelihood; the exp(posterior_score) is proportional to
# posterior probability, so we can estimate the normalizing constant Z like this:
Z = logsumexp([h.posterior_score for h in hypotheses])

for h in sorted(hypotheses, key=lambda h: h.posterior_score):
    print h.prior, h.likelihood, exp(h.posterior_score - Z), h



"""
## Added lot_iter so you can ctrl-c out
likelihoods = {}

for i in xrange(num_iters):
	t = NumberGameExpression(grammar)
	print t
	likelihoods[t] = t.compute_likelihood(data)

for t in likelihoods.keys():
	# use prior, likelihood to calculate posterior for hypothesis t 
	prior = t.compute_prior()
	likelihood = t.compute_likelihood(data)
	posterior = prior * likelihood

	# normalize
	normalize_param = sum(likelihoods.items() - likelihood)
	posterior = posterior / normalize_param
"""




from LOTlib import break_ctrlc

from LOTlib.Grammar import Grammar
from LOTlib.Hypotheses.LOTHypothesis import *
from LOTlib.Evaluation.Eval import *
from LOTlib.Miscellaneous import logsumexp, exp
from math import log
import matplotlib.pyplot as plt


#~~~~~~~~~~~~~~~~~~~~~~~~ 1 : Wrapper class for hypothesis ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
class NumberGameExpression(LOTHypothesis):

    def __init__(self, grammar, domain=100, noise=0.9, args=['n'], **kwargs):
        LOTHypothesis.__init__(self, grammar, args=args, **kwargs)
        self.domain = domain
        self.noise = noise

    def compute_likelihood(self, data):
        """ Likelihood of given data being produced by this hypothesis. """

        # Get domain subset that data would map to using hypothesis function
        subset = map(self, map(float, range(1, self.domain + 1)))
        subset = [item for item in subset if item <= self.domain]

        # The likelihood for each datum 'd' will depend on whether d is in subset
        s = len(subset)
        self.likelihood = 0

        for datum in data:
            if datum in subset:
                self.likelihood += log((self.noise /s) + ((1-self.noise) /self.domain))
            else:   # If datum not in hypo subset OR hypo subset is empty, use noise = (1-alpha)
                self.likelihood += log((1-self.noise) / self.domain)

        # This is required in all compute_likelihoods
        self.posterior_score = self.prior + self.likelihood
        return self.likelihood


#~~~~~~~~~~~~~~~~~~~~~~~~ 1b : Wrapper class for hypothesis ~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
class NumberGameExpression2(LOTHypothesis):

    def __init__(self, grammar, noise=0.9, args=[], **kwargs):
        LOTHypothesis.__init__(self, grammar, args=args, **kwargs)
        self.noise = noise

    def compute_likelihood(self, data):
        """
            Likelihood of specified data being produced by this hypothesis. If datum item
            not in set, it still has 'noise' likelihood of being generated.
        """
        h = self.__call__()     # Get hypothesis set
        alpha = self.noise
        noise = (1-alpha) / (max(h)-min(h))
        self.likelihood = 0
        for d in data:
            if d in h:
                self.likelihood += log(alpha/len(h) + noise)
            else:
                self.likelihood += log(noise)

        # This is required in all compute_likelihoods
        self.posterior_score = self.prior + self.likelihood
        return self.likelihood


"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2 : Create grammar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# - Rules in our grammar map between sets of integers in our domain                     #
# - E.g. "3n+1" maps to { 1, 4, 7, 10, ... }                                            #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
grammar = Grammar()
grammar.add_rule('START', '', ['EXPR'], 1)
# grammar.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], 1)
grammar.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], 1)
# grammar.add_rule('EXPR', 'minus_', ['EXPR', 'EXPR'], 1)
grammar.add_rule('EXPR', 'pow_', ['EXPR', 'EXPR'], 1)

# Converted to digits so python can evaluate them
grammar.add_rule('EXPR', 'n', None, 10)
for i in range(1, 3):
    grammar.add_rule('EXPR', str(i), None, (10-i)/2)


#~~~~~~~~~~~~~~~~~~~~~~~~ 3 : Generate hypotheses for data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Edit these parameters
num_iters = 50000
data = [3, 9, 17]
domain = 100
noise = 0.9
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
'''
from LOTlib.Inference.MetropolisHastings import MHSampler

hypotheses = set()
h0 = NumberGameExpression(grammar, domain=domain, noise=noise)
for h in MHSampler(h0, data, steps=num_iters):
    #print h   # with this you can see how hill climbing moves towards maxima
    hypotheses.add(h)
'''

print '\nGenerating ',num_iters,' hypotheses...\n'
hypotheses = set()
for _ in xrange(num_iters):       # You can ctrl+C to exit this loop
    t = NumberGameExpression(grammar, domain=domain, noise=noise)
    t.compute_posterior(data)               # Get prior, likelihood, posterior
    hypotheses.add(t)

# exp(posterior_score) is proportional to posterior probability, so we estimate normalizing
# constant Z like so:
Z = logsumexp([h.posterior_score for h in hypotheses])


#~~~~~~~~~~~~ printing prior, likelihood, & normalized posterior score ~~~~~~~~~~~~~~~~~#
best_hypotheses = sorted(hypotheses, key=lambda x: x.posterior_score)
best_hypotheses.reverse()
print '======================================='
for i in reversed(range(10)):
    h = best_hypotheses[i]
    print (i+1),':\t','Hypothesis:\t',h
    print '\tPosterior:\t%.5f' % exp(h.posterior_score - Z)
    # print '\t\tPrior:\t\t%.3f' % h.prior
    # print '\t\tLikehd:\t\t%.3f' % h.likelihood
    print '======================================='

print '\n'
print str(num_iters) + ' total hypotheses generated'
print str(len(hypotheses)) + ' unique hypotheses'


#~~~~~~~~~ graph p(y in C | d) as a function of y in domain (see figs 3,4,9) ~~~~~~~~~~~#
predictive_dist = [0.]*domain

'''
# METHOD 1 : calculate ???
for q in range(domain):
    # Sum p(q|h) * p(h|D) for all hypotheses h
    for h in hypotheses:
        posterior = h.posterior_score - Z
        likelihood = h.compute_likelihood([q+1])
        predictive_dist[q] += exp(posterior + likelihood)
'''

# METHOD 2 : calculate chance of sampling for each item in domain
for h in hypotheses:
    subset = map(h, map(float, range(1, domain + 1)))
    subset = [item for item in subset if item <= domain]
    for q in range(domain):
        posterior = h.posterior_score - Z
        predictive_dist[q] += exp(posterior) * (q in subset)




fig, ax = plt.subplots()
rects = plt.bar(range(1,domain+1), predictive_dist)

plt.xlabel('Domain (from 1 to '+str(domain)+')')
plt.ylabel('Probability of sampling this value')
plt.title('This graph needs a title!')

plt.show()



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ N O T E S ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# assuming you run MH long enough to get best hypotheses, Z should be pretty accurate

# it would be good to keep track of # of times you visit each state (hypothesis) during MH,
# so that we can compare that to using the posterior scores; after a lot of iterations we should
# see them converge

"""
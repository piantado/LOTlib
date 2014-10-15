__author__ = 'eric'
from Specification import *
from LOTlib import lot_iter
from LOTlib.Miscellaneous import logsumexp, exp
from LOTlib.Inference.MetropolisHastings import MHSampler

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Randomly generate samples ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def randomSample(grammar, data, num_iters=10000, alpha=0.9):
    hypotheses = set()
    for i in lot_iter(xrange(num_iters)):
        if i % 2000 == 0: print '\r\nGenerating %i hypotheses...\n' % i
        t = NumberSetHypothesis(grammar, alpha=alpha)
        t.compute_posterior(data)       # Get prior, likelihood, posterior
        hypotheses.add(t)
    return hypotheses


def mhSample(grammar, data, num_iters=10000, alpha=0.9):
    hypotheses = set()
    h0 = NumberSetHypothesis(grammar, alpha=alpha)
    for h in MHSampler(h0, data, steps=num_iters):
        #print h   # with this you can see how hill climbing moves towards maxima
        hypotheses.add(h)
    return hypotheses


# Estimate normalizing constant Z
def normalizingConstant(hypotheses):
    return logsumexp([h.posterior_score for h in hypotheses])


# Print top 10 hypotheses
def printBestHypotheses(hypotheses, n=10):
    best_hypotheses = sorted(hypotheses, key=lambda x: x.posterior_score)
    best_hypotheses.reverse()
    Z = normalizingConstant(hypotheses)

    print 'printing top '+str(n)+' hypotheses /  '+str(len(hypotheses))+' unique hypotheses\n'
    print '================================================================'
    for i in range(n):
        h = best_hypotheses[i].__call__()
        print (i+1), ' ~\t', 'Hypothesis:\t', str(h)
        print '\tPosterior:\t%.5f' % exp(h.posterior_score - Z)
        print '\t\tPrior:\t\t%.3f' % h.prior
        print '\t\tLikehd:\t\t%.3f' % h.likelihood
        print '================================================================\n'

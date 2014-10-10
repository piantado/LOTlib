__author__ = 'eric'
from Parameters import *
from NumberSetHypothesis import *
from pylab import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Initial parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Input data
# data = [1,1,1,3,3,3,3,3,5,5,7,7,9,9,11,11,11,15,15,17,17,17,17,19,19,19]  # for 2n - 1
DATA = [3, 9, 17]       # for 2^n + 1
# Number of hypotheses to generate
NUM_ITERS = 50000
# Noise parameter
ALPHA = 0.9




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Randomly generate samples ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def genHypotheses(grammar, data, num_iters=10000, alpha=0.9):
    hypotheses = set()
    for i in lot_iter(xrange(num_iters)):
        if i % 2000 == 0: print '\r\nGenerating %i hypotheses...\n' % i
        t = NumberSetHypothesis(grammar, alpha=alpha)
        t.compute_posterior(data)       # Get prior, likelihood, posterior
        hypotheses.add(t)

    return hypotheses


hypotheses = genHypotheses(grammar, NUM_ITERS, ALPHA, DATA)
Z = logsumexp([h.posterior_score for h in hypotheses])      # Estimate normalizing constant Z
### Will this be accurate??? h is recorded in a *set* not a list...

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Printing top 10 hypotheses ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
best_hypotheses = sorted(hypotheses, key=lambda x: x.posterior_score)
best_hypotheses.reverse()

print '\nGenerated %i hypotheses in total' % NUM_ITERS
print str(len(hypotheses)) + ' unique hypotheses\n'
print '================================================================'
for i in range(10):
    h = best_hypotheses[i].__call__()
    print (i+1),' ~\t','Hypothesis:\t',str(h)
    print '\tPosterior:\t%.5f' % exp(h.posterior_score - Z)
    print '\t\tPrior:\t\t%.3f' % h.prior
    print '\t\tLikehd:\t\t%.3f' % h.likelihood
    print '================================================================\n'

__author__ = 'eric'
from Shared import *
import NumberSetHypothesis
from pylab import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Initial parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Input data
# data = [1,1,1,3,3,3,3,3,5,5,7,7,9,9,11,11,11,15,15,17,17,17,17,19,19,19]  # for 2n - 1
data = [3, 9, 17]       # for 2^n + 1
# Number of hypotheses to generate
num_iters = 50000
# Noise parameter
alpha = 0.9




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Randomly generate samples ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Initialize an empty set to store our hypotheses
hypotheses = set()

# Iterate for the specified number of times
for _ in lot_iter(xrange(num_iters)):
    if _ % 2000 == 0:
        print '\r\nGenerating %i hypotheses...\n' % _                # Printing statement #

    # Generate a hypothesis & add it to our set
    t = NumberSetHypothesis(grammar, alpha=alpha)
    t.compute_posterior(data)                        # Get prior, likelihood, posterior #
    hypotheses.add(t)

# Estimate normalizing constant Z
Z = logsumexp([h.posterior_score for h in hypotheses])





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Printing top 10 hypotheses ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
best_hypotheses = sorted(hypotheses, key=lambda x: x.posterior_score)
best_hypotheses.reverse()

print '\nGenerated %i hypotheses in total' % num_iters
print str(len(hypotheses)) + ' unique hypotheses\n'
print '================================================================'

for i in range(10):
    h = best_hypotheses[i]
    # C_h is the set of items this hypothesis maps to; e.g. 2^n => {2,4,8,16,...}
    C_h = map(h, map(float, range(1, domain + 1)))
    C_h = [item for item in C_h if item <= domain]

    # Create figure
    figure(num=None, figsize=(8, 1.5), dpi=80, facecolor='w', edgecolor='k')
    hist(C_h, bins=domain, range=(1,domain))
    title('Hypothesis: '+str(h)+'   \nPosterior: %.5f' % exp(h.posterior_score - Z)
          +'\nPrior: %.3f' % h.prior + '\nLikehd: %.3f' % h.likelihood)

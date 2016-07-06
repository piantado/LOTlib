from __future__ import division
import pickle
from scipy.misc import logsumexp
import numpy as np

from Model import MyHypothesis
from LOTlib.DataAndObjects import FunctionData
print "Loading the hypothesis space . . ."
spaceset = pickle.load(open("topsybop.pkl", "r"))



#make me a list
space = list(spaceset)
for h in space:
    if(h.value.count_nodes() == 52):
       space.remove(h)
#print space[0].compute_likelihood([FunctionData(input=[], output={'b i m': 100, 'b o p': 100})])
#counts= space[0].make_ll_counts('bim')


def make_counts(hypothesis, n=50):
     counts={}
     for _ in xrange(n):
         val = hypothesis()
         if val not in counts:
            counts.update({val:1})
         else:
             counts[val]+=1
     return counts

#make a list of the posterior_scores
posteriors=[]
for h in space:
    posteriors.append(h.posterior_score)

#sum of the posterior scores gives you the probability of the data
pdata = logsumexp(posteriors)
print "P (D) : " + str(pdata)

#but we want to know P(H|D), the probability of each of these hypotheses given the data. We normalize to find out!
# (Now that we have a finite hypothesis space we can do this)

bipcounts={}
pbip = 0
for h in space:
    c = make_counts(h)

    if 'b i p' in c:
        prop = float((c['b i p'])/50) #proportion of seeing 'b i p' over the number of samples, which was 50 in make_counts
        logprop = np.log(prop)
        bipcounts.update({h:(logprop, h.posterior_score-pdata)}) #give a tuple of info: the log probability of bip for that hypothesis, and the probability of the hypothesis given the data
        pbip += logprop + (h.posterior_score-pdata)
        #print h, c


print bipcounts
print "Apprently, the probability of how much someone might expect to see 'bip' is exp("+ str(pbip) +")"










#Suppose I gave people data "bim" and "bop". What is the probability that they will expect to see "bip"?

#To be clear, that probability is a sum (marginalization) over all rules. So you need to find good, high probability rules,
# find the probability of each rule r, and see how likely r is to produce "bip". All with a nice interface.

#Yes, say you store the top 1000 hypotheses. Each has a posterior_score who is proportional to that hypothesis's log probability.
# So one way to do it is to take each hypothesis, generate strings, and weight the proportion of "bip" strings each hypothesis h generates
# by the probability of h being true given the data:

#P(bip | data) = sum_{h} P(h | data) P(bip | h)

#Just the definition of marginal probability.
# To compute p(H|data), you need to renormalize each h's posterior_score (be sure to use logsumexp)


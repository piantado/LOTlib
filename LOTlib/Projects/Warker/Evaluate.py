from __future__ import division
import pickle
from scipy.misc import logsumexp
import numpy as np
from LOTlib.Miscellaneous import Infinity, logplusexp, nicelog
from Model import MyHypothesis

print "Loading the hypothesis space . . ."
#load the hypothesis space
spaceset = pickle.load(open("tophyp.pkl", "r"))

#make me a list
space = list(spaceset)

#probably have to deal with this later, but why is this one too long? It can't be evaluated!!!
for h in space:
    if(h.value.count_nodes() >= 50):
       space.remove(h)



#make a list of the posterior_scores
posteriors=[]
for h in space:
    posteriors.append(h.posterior_score)

#sum of the posterior scores gives you the probability of the data
pdata = logsumexp(posteriors)

#but we want to know P(H|D), the probability of each of these hypotheses given the data. We normalize to find out!
# (Now that we have a finite hypothesis space we can do this)


#working with logs!
#pbip = -Infinity


illegals = ['s e k', 'N e k', 's e g', 'N e g', 's e m', ' N e m', 's e n', 'N e n', 'k e f', ' k e h', 'g e f', 'g e h', 'm e f', 'n e f', 'n e h']
illegal_probs = dict((w, -Infinity) for w in illegals)


for w in illegal_probs:
    illegal_probs[w] = np.exp(logsumexp([nicelog(h.ll_counts[w] + 1e-6) - nicelog(sum(h.ll_counts.values())+(1e-6*len(h.ll_counts.keys()))) + (h.posterior_score - pdata) for h in space]))

print illegal_probs.keys()
print illegal_probs.values()


#pbip = np.exp(logsumexp([nicelog(h.ll_counts['b i p']) - nicelog(sum(h.ll_counts.values())) + (h.posterior_score - pdata) for h in space]))
#print "Given 'bim' and 'bop', the probability that I will expect to see 'bip' is:  " + str(pbip)

legals = ['m e s', 'm e g', 'h e g', 'm e m', 'm e n', 'k e N', 'm e k', 'k e s', 'h e k', 'h e m', 'k e g', 'k e k', 'm e N', 'k e n', 'h e N', 'f e N', 'g e N', 'n e N', 'f e k', 'f e n', 'g e n', 'g e m', 'f e m', 'g e k', 'n e s', 'g e g', 'f e g', 'f e s', 'n e g', 'k e m', 'n e n', 'n e m', 'g e s', 'n e k']

stim = set(np.random.choice(legals,4))
legals = set(legals) - stim
print stim

legal_probs = legal_probs = dict((w, -Infinity) for w in legals)
print legal_probs

for w in legal_probs:
    legal_probs[w] = np.exp(logsumexp([nicelog(h.ll_counts[w] + 1e-6) - nicelog(sum(h.ll_counts.values())+(1e-6*len(h.ll_counts.keys()))) + (h.posterior_score - pdata) for h in space]))



# what I want to do: calculate percentages of legal errors
# make the space
# "present" a small subset of legal words (d)
# P(H|d) = P(d|H)P(H) <-- how do I update you?
# could use previous posterior as this new version's prior:
# P(H|d1) = P(d1|H)P(H|d0)






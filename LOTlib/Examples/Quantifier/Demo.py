# -*- coding: utf-8 -*-
"""
	Demo MCMC through lexica. Generally does not work well (too slow) so use the vectorized Gibbs version. 
"""
import re
from Shared import *

show_baseline_distribution()
print "\n\n"

# intialize a learner lexicon, at random
learner = GriceanSimpleLexicon(G, args=['A', 'B', 'S'])

for w in target.all_words():
	learner.set_word(w, G.generate('START')) # eahc word returns a true, false, or undef (None)

## sample the target data
data = generate_data(1000)

## Update the target with the data
target.compute_likelihood(data)

### Now we have built the data, so run MCMC
#for s in LOTlib.MetropolisHastings.mhgibbs_sample(learner, data, 100000, mh_steps=10, gibbs_steps=10):
#for s in LOTlib.MetropolisHastings.tempered_sample(learner, data, 1000, within_steps=10, temperatures=[1.0, 1.1], swaps=1):
#for s in LOTlib.MetropolisHastings.tempered_transitions_sample(learner, data, 1000, skip=0, temperatures=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]):
for s in mh_sample(learner, data, 10000, skip=0):
	
	sstr = str(s)
	sstr = re.sub("[_ ]", "", sstr)
	
	sstr = re.sub("presup", u"\u03BB A B . presup", sstr)
	
	print s.posterior_score, "\t", s.prior, "\t", s.likelihood, "\t", target.likelihood, "\n", sstr, "\n\n"
	
	

	
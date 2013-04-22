import psyco
psyco.full()

#from closure import *

from bvPCFG import *
from BasicPrimitives import *
from MetropolisHastings import *
from Likelihoods import *
from FiniteSample import *
from Miscellaneous import *
from Lexicon import *
from Objects import *

from random import randint

weights = range(1,10)
weights = map(lambda x: float(x)/sum(weights), weights)

#########################################################
#Define a class for running MH
class SimpleDiscreteSampler(Hypothesis):
	
	def __init__(self, i): 
		Hypothesis.__init__(self)
		self.i = i
		
		
	def __repr__(self): return str(self.i)
	def __hash__(self): return hash(self.i)
	def __cmp__(self, x): return cmp(self.i, x.i)
	
	def propose(self): 
		p = randint(0, len(weights)-1)
		return [SimpleDiscreteSampler(p), 0.0]
	def compute_prior(self):
		self.prior = log(weights[self.i])
		self.lp = self.prior + self.likelihood
		return self.prior
	def compute_likelihood(self, data): 
		self.likelihood = 0.0
		self.lp = self.prior + self.likelihood
		return self.likelihood
		
q = SimpleDiscreteSampler(1)
d = dict()
#for s in BasicMCMC.tempered_transitions_sample(q, [], 5000, skip=0):		
#for s in BasicMCMC.tempered_sample(q, [], 5000, temperatures=[1.0, 2.0, 3.0]):
for s in BasicMCMC.mh_sample(q, [], 50000, skip=0):
	#print s.lp
#for x in range(1000):
	#i = weighted_sample(range(len(weights)), probs=weights)
	#s = Sample(i, log(weights[i]), 0.0)
	hashplus(d, s)#print ph

test_expected_counts(d)


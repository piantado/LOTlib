"""

	A class for representing sets of hypotheses. This extends list and gives us functions like
	computing posteriors, normalizing, posterior predictives, etc. We use list instead of set so 
	we can iterate in order
	
	This uses each hypothesis' .lp slot to store and update things
"""

#from LOTlib.FiniteBestSet import FiniteBestSet
#from LOTlib.Miscellaneous import *

#class HypothesisSet(list):
	
	#def __init__(hyps):
		#for h in hyps: self.add(h, h.lp)
	
	#def add(*h): self.extend(h)
		
	#def normalize():
		#self.Z = logsumexp([x.lp for x in self])
		
		## change in the Q -- should be okay since we alter everything
		#for h in self: h = h.lp - self.Z 
			
	#def compute_posterior(data):
		#for h in self: h.compute_posterior(data)
		
	#def posterior_predictive(data):
		#pass
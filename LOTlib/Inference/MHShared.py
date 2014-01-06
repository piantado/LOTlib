from collections import defaultdict

class MHStats(defaultdict):
	def __init__(self):
		defaultdict.__init__(self, int)
	
	def acceptance_ratio(self):
		if self.get('total') > 0:
			return float(self.get('accept',0)) / float(self.get('total',1))
		else:   return None

from math import log, exp, isnan
from random import random

def MH_acceptance(cur, prop, fb, acceptance_temperature=1.0):
	"""
		Handle all the weird corner cases for computing MH acceptance ratios
		And then returns a true/false for acceptance
	"""
	
	if isnan(cur) or (cur==float("-inf") and prop==float("-inf")): # if we get infs or are in a stupid state, let's just sample from the prior so things don't get crazy
		r = -log(2.0) #  just choose at random -- we can't sample priors since they may be -inf both
	elif isnan(prop): #never accept
		r = float("-inf")
	else:
		r = (prop-cur-fb) / acceptance_temperature
	
	# And flip
	return (r >= 0.0 or random() < exp(r))
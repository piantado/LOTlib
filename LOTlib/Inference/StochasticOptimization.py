# -*- coding: utf-8 -*-

"""
	This uses a simple stochastic optimization scheme, that cycles through data, 
	updating a sample a few times based only on *one* data point. This may help
	get out of local maxima

"""

from random import random
from math import log, exp, isnan

import LOTlib
from LOTlib.Miscellaneous import *
from LOTlib.FiniteBestSet import FiniteBestSet
from MetropolisHastings import mh_sample
from MHShared import *

def datawise_optimize(current_sample, data, steps=1000000, inner_steps=10, data_weight=1.0, ll_temperature=1.0, **kwargs):
	"""
		cycle through data points, taking a few steps in the direction of that data point
		This uses ll_temperature to simulate having len(data)*data_weight number of data points
		
		steps -- you take this many total steps (steps/inner_steps inner loops)
		inner steps -- how many steps to take on a single data point
		data_weight -- weight each single data point as len(data)*this
		
		
	"""
	
	# How many data points? Used for setting the temperature below
	NDATA = len(data)
	
	for mhi in xrange(steps/inner_steps):
		if LOTlib.SIG_INTERRUPTED: break
	
		for di in data:
			if LOTlib.SIG_INTERRUPTED: break
			
			for h in mh_sample(current_sample, [di], steps=inner_steps, ll_temperature=NDATA*ll_temperature*data_weight, **kwargs):
				current_sample = h 
				yield h
				


	
if __name__ == "__main__":
	
	from LOTlib.Examples.Number.Shared import *
	
	data = generate_data(200)

	# A starting hypothesis (later ones are created by .propose, called in LOTlib.MetropolisHastings
	initial_hyp = NumberExpression(G)

	for h in datawise_optimize(initial_hyp, data):
		#pass
		print h.lp, "\t", h
	

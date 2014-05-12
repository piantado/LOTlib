"""
	Routines for evaluating MCMC runs
	
	TODO:
	
		MAYBE USE: #LOTlib.Hypothesis.POSTERIOR_CALL_COUNTER and report how many calls we've made
"""
import LOTlib
from LOTlib import lot_iter
from SimpleMPI.ParallelBufferedIO import ParallelBufferedIO
from LOTlib.Miscellaneous import *
from collections import defaultdict
from math import log, exp
from scipy.stats import chisquare
import numpy
from time import time

def evaluate_sampler(target, sampler, print_every=250, steps=1000000, chains=1, name="", trace=False, output=sys.stdout):
	"""
		target - a hash from hypotheses to lps. The keys of this are the only things we 
		sampler - a sampler we wish to evaluate_sampler
		print_every -- print out every this many samples
		steps -- how many total samples to draw
		prefix -- anything to print before
		trace - should we print a hypothesis every step?
		assertion - assertion fail if the chi squared doesn't work out
		chains - how many outside chains to run?
		
		print a trace of stats:
			- prefix
			- what chain
			- how many samples
			- KL (via counts)
			- percent of hypotheses found
			- percent of probability mass found
			- number of target 
			- length of found samples
			- target normalizer (log)
			- count of samples overlapping with target
			- count of samples NOT overlapping with target
			- chi squared statistic
			- p value
			
		NOTE: You may get no output if there isn't any overlap between samples and target
	"""
	
	hypotheses = target.keys()
	tZ = logsumexp(target.values()) # get the normalizer
	
	for chain_i in lot_iter(xrange(chains)):
		samples = defaultdict(int) # keep track of samples
		
		startt = time()
		for n, s in lot_iter(enumerate(sampler)): # each sample should have an .lp defined
			
			samples[s] += 1
			
			if trace: print "#", n, s in target, s.lp, s
			
			if (n%print_every)==0 and n>0:
				
				# the sum of everything in hypotheses
				sm = sum( [samples[x] for x in hypotheses ] )
				if sm == 0: continue
				
				# the counts of things outside of hypotheses
				sm_out = n - sm 
				sZ = log(sm)
				
				if sm > 0:
					KL = 0.0
				
					for h in hypotheses:
						Q = target[h] - tZ
						sh = samples[h]
						if sh > 0:
							P = log( sh ) - sZ
							KL += (P - Q) * exp( P )
				else:
					KL = float("inf")
					
				# And compute the percentage found
				percent_found = float(sum([ (samples[x]>0) for x in hypotheses]))/ float(len(hypotheses))
				pm_found = logsumexp([target[x] for x in hypotheses if samples[x] > 0])
								
				# compute chi squared counts
				#fobs = numpy.array( [samples[h] for h in hypotheses] )
				#fexp = numpy.array( [ numpy.exp(target[h]-tZ) * sm for h in hypotheses])
				#chi,p = chisquare(fobs, f_exp=fexp)  ## TODO: check ddof
				
				output.write('\t'.join(map(str, [name, chain_i, n, time()-startt, r3(KL), r3(percent_found), r4(pm_found-tZ), len(hypotheses), len(samples.keys()), r4(tZ)]  )) + '\n')
			
			if n > steps: break
	
"""
	Routines for evaluating MCMC runs
	
	TODO:
	
		MAYBE USE: #LOTlib.Hypothesis.POSTERIOR_CALL_COUNTER and report how many calls we've made
"""
import LOTlib
from SimpleMPI.ParallelBufferedIO import ParallelBufferedIO
from LOTlib.Miscellaneous import *
from collections import defaultdict
from math import log, exp
from scipy.stats import chisquare
import numpy
from time import time

def evaluate_sampler(target, sampler, print_every=1000, steps=1000000, chains=1, prefix="", trace=False, outfile=None, assertion=False):
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
	
	# If we write to an output (in parallel)
	if outfile is not None: 
		bo = ParallelBufferedIO(outfile)
	
	for chain_i in xrange(chains):
		samples = defaultdict(int) # keep track of samples
		
		startt = time()
		for n, s in enumerate(sampler): # each sample should have an .lp defined
			if LOTlib.SIG_INTERRUPTED: break
			
			samples[s] += 1
			
			if trace: 
				print "#", n, s in target, s.lp, s
			
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
				fobs = numpy.array( [samples[h] for h in hypotheses] )
				fexp = numpy.array( [ numpy.exp(target[h]-tZ) * sm for h in hypotheses])
				chi,p = chisquare(fobs, f_exp=fexp)  ## TODO: check ddof
				
				if outfile is None:
					print prefix, chain_i, n, r3(KL), r3(percent_found), r4(exp(pm_found-tZ)), len(hypotheses), len(samples.keys()), r4(tZ), sm, sm_out, r3(chi), r3(p)
				else:
					bo.write(prefix, chain_i, n, r3(KL), r3(percent_found), r4(exp(pm_found-tZ)), len(hypotheses), len(samples.keys()), r4(tZ), sm, sm_out, r3(chi), r3(p))
		
			if n > steps: break
		
		if outfile is not None: bo.close()
		return 

## this take sa dictionary d
## the keys of d must contain "lp", and d must contain counts
## this prints out a chi-squared test to make sure these are right
## NOTE: This doe snot do well if we have a fat tail, since we will necessarily sample some low probability events
##from scipy.stats import chisquare
### importantly, throw out counts less than min_count -- else we get crummy
#def test_expected_counts(d, display=True, sort=True, min_count=100):
	#keys = d.keys() # maintain an order for the keys
	#if sort:
		#keys = sorted(keys, key=lambda x: d[x])
	#lpZ = logsumexp([ k.lp for k in keys])
	#cntZ = sum(d.values())
	#if display:			
		#for k in keys:
			#ocnt = float(d[k])/cntZ
			#ecnt = exp(k.lp-lpZ)
			#print d[k], "\t", ocnt, "\t", ecnt, "\t", ocnt/ecnt, "\t", k
	## now update these with their other probs	
	#keeper_keys = filter(lambda x: d[x] >= min_count, keys)
	##print len( keeper_keys), len(keys), map(lambda x: d[x], keys)
	#lpZ = logsumexp([ k.lp for k in keeper_keys])
	#cntZ = sum([d[k] for k in keeper_keys])	
	
	## The chisquared test does not do well here iwth the low expected counts -- 
	#print chisquare( [ d[k] for k in keeper_keys ], f_exp=array( [ cntZ * exp(k.lp - lpZ) for k in keeper_keys] ))  ##UGH expected *counts*, not probs


	



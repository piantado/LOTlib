"""
	Routines for evaluating MCMC runs
	
	TODO:
	
		Jan 12 2013 -- Hmm things aren't working so well yet... It appears that maybe it doesn't accurately detect when it's found something it already has??
"""

from LOTlib.Miscellaneous import *
from collections import defaultdict
from math import log, exp
from scipy.stats import chisquare
import numpy

def evaluate_sampler(target, sampler, skip=10000, steps=10000000, chains=1, prefix="", trace=False, outfile=None, assertion=False):
	"""
		target - a hash from hypotheses to lps. The keys of this are the only things we 
		sampler - a sampler we wish to evaluate_sampler
		skip -- print out every this many samples
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
	"""
	
	hypotheses = target.keys()
	tZ = logsumexp(target.values()) # get the normalizer
	if outfile is not None: bo = ParallelBufferedIO(outfile)
	
	for chain_i in xrange(chains):
		samples = defaultdict(int) # keep track of samples
		
		n = 0
		for s in sampler: # each sample should have an .lp defined
			
			samples[s] += 1
			n += 1
			
			if trace: print "#", s in target, s.lp, s
			
			if (n%skip)==0:
				
				sm = sum( [samples[x] for x in hypotheses ] )
				if sm == 0: continue
				
				sm_out = sum(samples.values()) - sm # the counts of things outside of hypotheses
				sZ = log(sm)
				
				KL = 0.0
				for h in hypotheses:
					Q = target[h] - tZ
					sh = samples[h]
					if sh > 0:
						P = log( sh ) - sZ
						KL += (P - Q) * exp( P ) # otherwise, limit->0
					
				# And compute the percentage found
				percent_found = float(sum([1 for x in hypotheses if samples[x] > 0]))/ float(len(hypotheses))
				pm_found = logsumexp([target[x] for x in hypotheses if samples[x] > 0])
				
				for k in hypotheses: 
					if samples[k] == 0: print k
				
				# compute chi squared counts
				fobs = numpy.array( [samples[h] for h in hypotheses] )
				fexp = numpy.array( [ numpy.exp(target[h]-tZ) * sm for h in hypotheses])
				chi,p = chisquare(fobs, f_exp=fexp)  
				
				if outfile is None:
					print prefix, chain_i, n, r3(KL), r3(percent_found), r3(pm_found), len(hypotheses), len(samples.keys()), r3(tZ), sm, sm_out, r3(chi), r3(p)
				else:
					bo.write(prefix, chain_i, n, r3(KL), r3(percent_found), r3(pm_found), len(hypotheses), len(samples.keys()), r3(tZ), sm, sm_out, r3(chi), r3(p))
		
			if n > steps: break
		
		if outfile is not None: bo.close()
		return 

	
	
	
	
	
"""
  ########################################################################################################################
  ## Evaluation of different samplers
  ########################################################################################################################
"""

#def compute_sampler_performance(sampler, target, nsamples=xrange(0,1000,100), maxcalls=float("+inf"), pre="", TRACE=False):
	#"""
		#Sampler here is a generator; Target is a PriorityQueue
		
	#"""
	
	#N = len(target) # the target distribution
	#targethyps = target.get_sorted(decreasing=True)
	#targetset = set(targethyps)
	#Z = logsumexp([ x.lp for x in targethyps ])
	##print targethyps[0:25]
	
	#mysamples = FiniteBestSet(max=True, N=N)
	
	
	## reset this
	#LOTlib.Hypothesis.POSTERIOR_CALL_COUNTER = 0
	
	#cnt = 0
	#total_time = 0.0
	#tick = time.time()
	#for s in sampler:
		##if TRACE: print s
		#total_time += (time.time() - tick)
		
		#if cnt > max(nsamples) or LOTlib.Hypothesis.POSTERIOR_CALL_COUNTER > maxcalls: 
			#break # *slightly* inefficient
		
		#if cnt in nsamples: # *slightly* inefficient
			
			## print out at these amounts
			#ms = mysamples.get_sorted(decreasing=True)
			#myZ = logsumexp([ x.lp for x in ms ])
			#myS = set(ms)
			#topN = 0 # how many in order did we get right?
			#for x in targethyps:
				#if x in myS: topN += 1
				#else:        break
				
			#print pre, cnt, total_time, LOTlib.Hypothesis.POSTERIOR_CALL_COUNTER,  myZ, Z, len(myS & targetset), topN, N, q(ms[0])
			
		#mysamples.push(s, s.lp)	
		#cnt += 1
		#tick = time.time() # update this - we only want to measure the time spent generating
		
	
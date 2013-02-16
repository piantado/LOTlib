"""
	Routines for evaluating MCMC runs
	
	TODO:
	
		Jan 12 2013 -- Hmm things aren't working so well yet... It appears that maybe it doesn't accurately detect when it's found something it already has??
		
		-- Make Target a hash so we can actually put things in it!!
"""

from LOTlib.Miscellaneous import *
from collections import defaultdict
from math import log, exp
from scipy.stats import chisquare
import numpy

def evaluate_sampler(target, sampler, skip=100, steps=100000, prefix="", trace=False, outfile=None):
	"""
		target - a hash from hypotheses to lps. The keys of this are the only things we 
		sampler - a sampler we wish to evaluate_sampler
		
		print a trace of stats:
			- KL(sample || target) -- KL estimate
			- KL(sample || target) -- seen hypothesis estimate
			- percentage of hypotheses found
			- Total probability mass found
	"""
	
	hypotheses = target.keys()
	tZ = logsumexp(target.values()) # get the normalizer
	
	# keep track of samples
	posterior_samples = defaultdict(int)
	
	if outfile is not None: bo = ParallelBufferedIO(outfile)
	
	n = 0
	for s in sampler:
		
		if s in target: 
			posterior_samples[s] += 1
			#print ">>", s
		if trace: print "#", s in target, s.lp, s
		
		if (n%skip)==(skip-1):
			
			sm = sum( [posterior_samples[x] for x in posterior_samples.keys() ] )
			
			if sm == 0: print "# No valid samples yet for evaluate_sampler"
			else:
				sZ_lp = logsumexp( [x.lp for x in posterior_samples.keys()] )
				sZ_cnt = log(sm)
				
				KL_lp = 0.0
				KL_cnt = 0.0
				for h,cnt in posterior_samples.items():
					Q = target[h] - tZ
					
					P_lp = h.lp - sZ_lp # log prob estimate for h's probability
					KL_lp += (P_lp - Q) * exp( P_lp )
					
					P_cnt = log(cnt) - sZ_cnt
					KL_cnt += (P_cnt - Q) * exp(P_cnt)
				
				# And compute the percentage found
				percent_found = float(len( set(posterior_samples.keys()) & set(hypotheses) )) / float(len(hypotheses))
				
				# compute chi squared counts
				fobs = numpy.array( [posterior_samples.get(h,0) for h in hypotheses] )
				fexp = numpy.array( [ numpy.exp(h.lp-tZ) * len(hypotheses) for h in hypotheses])
				chi,p = chisquare(fobs, f_exp=fexp)  
				
				if outfile is None:
					print prefix, n, KL_lp, KL_cnt, percent_found, len(hypotheses), len(posterior_samples.keys()), tZ, sZ_lp, log(sZ_cnt), chi, p
				else:
					bo.write(prefix, n, KL_lp, KL_cnt, percent_found, len(hypotheses), len(posterior_samples.keys()), tZ, sZ_lp, log(sZ_cnt), chi, p)
		
		n += 1		
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
		
	
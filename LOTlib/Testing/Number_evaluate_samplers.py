# -*- coding: utf-8 -*-


"""
	Can run on mpi cluster as:
	time mpiexec -outfile-pattern evaluation-data/eval.%r --hostfile ../../hosts.mpich2 -n 23 python Number_evaluate_samplers.py
"""

from LOTlib.Examples.Number.Shared import *
from LOTlib.Testing.Evaluation import evaluate_sampler
from copy import copy
import sys

# We need to do this so that we can load via pickle (it searches for Shared)
sys.path.append("/home/piantado/Desktop/mit/Libraries/LOTlib/LOTlib/Examples/Number/") 

generate_target = False

DATA_SIZE = 100

TARGET_SAMPLES = 50000

TEST_SAMPLES = 10000
RUNS = 1000

#TARGET_FILE = "mpi-run.pkl" # load this from \
TARGET_FILE = "/home/piantado/Desktop/mit/Libraries/LOTlib/LOTlib/Examples/Number/mpirun-Dec2013.pkl" # load a small file. The large one is only necessary if we want the "correct" target likelihood and top N numbers; if we just look at Z we don't need it!
#TARGET_FILE = "/tmp/mpi-run.pkl" # load this from 
DATA_FILE = "data/evaluation-data.pkl"


if generate_target:	
	
	# make and save the data
	data = generate_data(DATA_SIZE)
	pickle_save(data, DATA_FILE)
	
	initial_hyp = NumberExpression(G)
	
	q = FiniteBestSet(10000)
	for h in LOTlib.MetropolisHastings.mh_sample(initial_hyp, data, TARGET_SAMPLES, skip=0):
		q.push(h, h.lp)
	q.save(TARGET_FILE)
	
	
else:
	
	data   = pickle_load(DATA_FILE)
	
	# recompute the target posterior in case it's diff data than was generated
	# target here must be a dict from hypotheses to posteriors
	target = dict()
	for h in pickle_load(TARGET_FILE).get_all():
		target[h] = sum(h.compute_posterior(data)) # add up the components returned by compute_posterior
	
	print "# Starting runs"
	for r in xrange(RUNS):
		if LOTlib.SIG_INTERRUPTED: break
		
		h0 = NumberExpression(G)
		
		#sampler = Number_enumerative_search(data, PRIOR_SCORE_TEMPERATURE=1.0, PRIOR_LIKELIHOOD_TEMPERATURE=float("+inf"), N=10000, breakout=1000, yield_immediate=False)
		#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="EnumerativeSearch-Prior\t"+str(r) )
		
		#sampler = Number_enumerative_search(data, PRIOR_SCORE_TEMPERATURE=1.0, PRIOR_LIKELIHOOD_TEMPERATURE=1.0, N=10000, breakout=1000, yield_immediate=False)
		#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="EnumerativeSearch-Posterior\t"+str(r) )
		
		#sampler = Number_enumerative_search(data, PRIOR_SCORE_TEMPERATURE=1.0, PRIOR_LIKELIHOOD_TEMPERATURE=5.0, N=10000, breakout=1000, yield_immediate=False)
		#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, LLS, prefix="EnumerativeSearch-LL5\t"+str(r) )
		
		sampler = LOTlib.MetropolisHastings.mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0)
		evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="BasicSampler\t"+str(r), trace=False )

		sampler = LOTlib.MetropolisHastings.mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0, temperature=1.01)
		evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="BasicSamplerT1.01\t"+str(r), trace=False )
		
		sampler = LOTlib.MetropolisHastings.mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0, temperature=1.05)
		evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="BasicSamplerT1.05\t"+str(r), trace=False )
		
		sampler = LOTlib.MetropolisHastings.mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0, temperature=1.1)
		evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="BasicSamplerT1.1\t"+str(r), trace=False )
		
		sampler = LOTlib.MetropolisHastings.mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0, temperature=1.5)
		evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="BasicSamplerT1.5\t"+str(r), trace=False )
		
		#sampler = LOTlib.MetropolisHastings.tempered_transitions_sample(copy(h0), data, TEST_SAMPLES, skip=0, temperatures=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
		#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="TemperedTransitions-1.5\t"+str(r) )
		
		#sampler = LOTlib.MetropolisHastings.tempered_transitions_sample(copy(h0), data, TEST_SAMPLES, skip=0, temperatures=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
		#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="TemperedTransitions-2.0\t"+str(r) )
		
		#sampler = LOTlib.MetropolisHastings.mhgibbs_sample(copy(h0), data, TEST_SAMPLES, mh_steps=25, gibbs_steps=1)
		#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="MHGibbs\t"+str(r) )
		
		#sampler = LOTlib.MetropolisHastings.parallel_tempering_sample(copy(h0), data, TEST_SAMPLES, within_steps=10, temperatures=(1.0, 1.05, 1.10), swaps=1)
		#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="ParallelTempring-1-1.05-1.1\t"+str(r) )
		
		#sampler = LOTlib.MetropolisHastings.parallel_tempering_sample(copy(h0), data, TEST_SAMPLES, within_steps=10, temperatures=(1.0, 1.25, 1.5), swaps=1)
		#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="ParallelTempring-1-1.25-1.5\t"+str(r) )


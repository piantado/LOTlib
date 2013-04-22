# -*- coding: utf-8 -*-


"""
	Can run on mpi cluster as:
	time mpiexec -outfile-pattern evaluation-data/eval.%r --hostfile ../../hosts.mpich2 -n 23 python Number_evaluate_samplers.py
"""

from Number_Shared import *
from Number_enumerative_search import *

generate_target = False

DATA_SIZE = 30
TARGET_SAMPLES = 5000000
TOP_N = 10000
TEST_SAMPLES = range(1, 1000000, 100)
MAX_CALLS = 10000000
RUNS = 1000

#TARGET_FILE = "mpi-run.pkl" # load this from \
TARGET_FILE = "mpi-run-small.pkl" # load a small file. The large one is only necessary if we want the "correct" target likelihood and top N numbers; if we just look at Z we don't need it!
#TARGET_FILE = "/tmp/mpi-run.pkl" # load this from 
DATA_FILE = "evaluation-data/evaluation-data.pkl"


if generate_target:	
	
	# make and save the data
	data = generate_data(DATA_SIZE)
	pickle_save(data, DATA_FILE)
	
	initial_hyp = NumberExpression(G)
	
	q = UniquePriorityQueue(max=True, N=TOP_N)
	for h in LOTlib.MetropolisHastings.mh_sample(initial_hyp, data, TARGET_SAMPLES, skip=0):
		q.push(h, h.lp)
	q.save(TARGET_FILE)
	
	
else:
	
	data   = pickle_load(DATA_FILE)
	
	# recompute the target posterior in case it's diff data than was generated
	target = pickle_load(TARGET_FILE)
	for h in target.get_all():
		h.compute_posterior(data)
	
	print "# Starting runs"
	for r in xrange(RUNS):
			
		initial_hyp = NumberExpression()
		
		sampler = Number_enumerative_search(data, PRIOR_SCORE_TEMPERATURE=1.0, PRIOR_LIKELIHOOD_TEMPERATURE=float("+inf"), N=10000, breakout=1000, yield_immediate=False)
		LOTlib.MetropolisHastings.compute_sampler_performance(sampler, target, nsamples=TEST_SAMPLES, maxcalls=MAX_CALLS, pre="EnumerativeSearch-Prior\t"+str(r) )
		
		sampler = Number_enumerative_search(data, PRIOR_SCORE_TEMPERATURE=1.0, PRIOR_LIKELIHOOD_TEMPERATURE=1.0, N=10000, breakout=1000, yield_immediate=False)
		LOTlib.MetropolisHastings.compute_sampler_performance(sampler, target, nsamples=TEST_SAMPLES, maxcalls=MAX_CALLS, pre="EnumerativeSearch-Posterior\t"+str(r) )
		
		sampler = Number_enumerative_search(data, PRIOR_SCORE_TEMPERATURE=1.0, PRIOR_LIKELIHOOD_TEMPERATURE=5.0, N=10000, breakout=1000, yield_immediate=False)
		LOTlib.MetropolisHastings.compute_sampler_performance(sampler, target, nsamples=TEST_SAMPLES, maxcalls=MAX_CALLS, pre="EnumerativeSearch-LL5\t"+str(r) )
		
		#sampler = LOTlib.MetropolisHastings.mh_sample(initial_hyp, data, max(TEST_SAMPLES), skip=0)
		#LOTlib.MetropolisHastings.compute_sampler_performance(sampler, target, nsamples=TEST_SAMPLES, maxcalls=MAX_CALLS, pre="BasicSampler\t"+str(r) )
		
		#sampler = LOTlib.MetropolisHastings.tempered_transitions_sample(initial_hyp, data, max(TEST_SAMPLES), skip=0, temperatures=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
		#LOTlib.MetropolisHastings.compute_sampler_performance(sampler, target, nsamples=TEST_SAMPLES, maxcalls=MAX_CALLS, pre="TemperedTransitions-1.5\t"+str(r) )
		
		#sampler = LOTlib.MetropolisHastings.tempered_transitions_sample(initial_hyp, data, max(TEST_SAMPLES), skip=0, temperatures=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
		#LOTlib.MetropolisHastings.compute_sampler_performance(sampler, target, nsamples=TEST_SAMPLES, maxcalls=MAX_CALLS, pre="TemperedTransitions-2.0\t"+str(r) )
		
		#sampler = LOTlib.MetropolisHastings.mhgibbs_sample(initial_hyp, data, max(TEST_SAMPLES), mh_steps=25, gibbs_steps=1)
		#LOTlib.MetropolisHastings.compute_sampler_performance(sampler, target, nsamples=TEST_SAMPLES, maxcalls=MAX_CALLS, pre="MHGibbs\t"+str(r) )
		
		#sampler = LOTlib.MetropolisHastings.parallel_tempering_sample(initial_hyp, data, max(TEST_SAMPLES), within_steps=10, temperatures=(1.0, 1.05, 1.10), swaps=1)
		#LOTlib.MetropolisHastings.compute_sampler_performance(sampler, target, nsamples=TEST_SAMPLES, maxcalls=MAX_CALLS, pre="ParallelTempring-1-1.05-1.1\t"+str(r) )
		
		#sampler = LOTlib.MetropolisHastings.parallel_tempering_sample(initial_hyp, data, max(TEST_SAMPLES), within_steps=10, temperatures=(1.0, 1.25, 1.5), swaps=1)
		#LOTlib.MetropolisHastings.compute_sampler_performance(sampler, target, nsamples=TEST_SAMPLES, maxcalls=MAX_CALLS, pre="ParallelTempring-1-1.25-1.5\t"+str(r) )


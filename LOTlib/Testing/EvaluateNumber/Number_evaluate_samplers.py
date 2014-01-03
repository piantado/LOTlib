# -*- coding: utf-8 -*-


"""
	Can run on mpi cluster as:
	
	mpiexec -n 8 python Number_evaluate_samplers.py
"""

from LOTlib.Examples.Number.Shared import *
from LOTlib.Testing.Evaluation import evaluate_sampler
from LOTlib.sandbox.ProbTaboo import ptaboo_search
from SimpleMPI.MPI_map import MPI_map

from copy import copy
import sys

# We need to do this so that we can load via pickle (it searches for Shared)
sys.path.append("/home/piantado/Desktop/mit/Libraries/LOTlib/LOTlib/Examples/Number/") 

DATA_SIZE = 400

TEST_SAMPLES = 10000
RUNS = 1000

TARGET_FILE = "/home/piantado/Desktop/mit/Libraries/LOTlib/LOTlib/Examples/Number/mpirun-Dec2013.pkl" # load a small file. The large one is only necessary if we want the "correct" target likelihood and top N numbers; if we just look at Z we don't need it!
DATA_FILE = "data/evaluation-data.pkl"
outfile = "evaluation.txt"

data   = pickle_load(DATA_FILE)

# recompute the target posterior in case it's diff data than was generated
# target here must be a dict from hypotheses to posteriors
target = dict()
for h in pickle_load(TARGET_FILE).get_all():
	target[h] = sum(h.compute_posterior(data)) # add up the components returned by compute_posterior

## A wrapper function for MPI_map
def run_one(r):
	h0 = NumberExpression(G)
	
	sampler = ptaboo_search( copy(h0), data, steps=TEST_SAMPLES, skip=0, seen_penalty=1.0)
	evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="PtabooSearch-1.0\t"+str(r), trace=False, outfile=outfile )

	sampler = ptaboo_search( copy(h0), data, steps=TEST_SAMPLES, skip=0, seen_penalty=10.0)
	evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="PtabooSearch-10.0\t"+str(r), trace=False, outfile=outfile )

	sampler = ptaboo_search( copy(h0), data, steps=TEST_SAMPLES, skip=0, seen_penalty=100.0)
	evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="PtabooSearch-100.0\t"+str(r), trace=False, outfile=outfile )
	
	sampler = LOTlib.MetropolisHastings.mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0)
	evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="BasicSampler\t"+str(r), trace=False, outfile=outfile)
	
	sampler = LOTlib.MetropolisHastings.mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0, temperature=1.01)
	evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="BasicSamplerT1.01\t"+str(r), trace=False, outfile=outfile )
	
	sampler = LOTlib.MetropolisHastings.mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0, temperature=1.05)
	evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="BasicSamplerT1.05\t"+str(r), trace=False, outfile=outfile )
	
	sampler = LOTlib.MetropolisHastings.mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0, temperature=1.1)
	evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="BasicSamplerT1.1\t"+str(r), trace=False, outfile=outfile )
	

# Actually run, in parallel!
MPI_map( run_one, range(RUNS) ) 


"""
#print "# Starting runs"
#for r in xrange(RUNS):
	#if LOTlib.SIG_INTERRUPTED: break
	
	#h0 = NumberExpression(G)
	
	#sampler = Number_enumerative_search(data, PRIOR_SCORE_TEMPERATURE=1.0, PRIOR_LIKELIHOOD_TEMPERATURE=float("+inf"), N=10000, breakout=1000, yield_immediate=False)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="EnumerativeSearch-Prior\t"+str(r) )
	
	#sampler = Number_enumerative_search(data, PRIOR_SCORE_TEMPERATURE=1.0, PRIOR_LIKELIHOOD_TEMPERATURE=1.0, N=10000, breakout=1000, yield_immediate=False)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="EnumerativeSearch-Posterior\t"+str(r) )
	
	#sampler = Number_enumerative_search(data, PRIOR_SCORE_TEMPERATURE=1.0, PRIOR_LIKELIHOOD_TEMPERATURE=5.0, N=10000, breakout=1000, yield_immediate=False)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, LLS, prefix="EnumerativeSearch-LL5\t"+str(r) )
	
	
	#sampler = ptaboo_search( copy(h0), data, steps=TEST_SAMPLES, skip=0, seen_penalty=1.0)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="PtabooSearch-1.0\t"+str(r), trace=False )

	#sampler = ptaboo_search( copy(h0), data, steps=TEST_SAMPLES, skip=0, seen_penalty=10.0)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="PtabooSearch-10.0\t"+str(r), trace=False )

	#sampler = ptaboo_search( copy(h0), data, steps=TEST_SAMPLES, skip=0, seen_penalty=100.0)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="PtabooSearch-100.0\t"+str(r), trace=False )
	
	#sampler = LOTlib.MetropolisHastings.mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="BasicSampler\t"+str(r), trace=False )
	
	

	#sampler = LOTlib.MetropolisHastings.mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0, temperature=1.01)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="BasicSamplerT1.01\t"+str(r), trace=False )
	
	#sampler = LOTlib.MetropolisHastings.mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0, temperature=1.05)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="BasicSamplerT1.05\t"+str(r), trace=False )
	
	#sampler = LOTlib.MetropolisHastings.mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0, temperature=1.1)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="BasicSamplerT1.1\t"+str(r), trace=False )
	
	#sampler = LOTlib.MetropolisHastings.mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0, temperature=1.5)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, prefix="BasicSamplerT1.5\t"+str(r), trace=False )
	
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

"""
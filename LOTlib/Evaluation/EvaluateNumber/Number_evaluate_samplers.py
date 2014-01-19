# -*- coding: utf-8 -*-


"""
	Can run on mpi cluster as:
	
	mpiexec -n 8 python Number_evaluate_samplers.py
"""

from LOTlib.Examples.Number.Shared import *
from LOTlib.Evaluation.Evaluation import evaluate_sampler
from LOTlib.Inference.StochasticOptimization import datawise_optimize
from LOTlib.Inference.MetropolisHastings import mh_sample
from LOTlib.Inference.ProbTaboo import ptaboo_search
from LOTlib.Inference.IncreaseTemperatureMH import increase_temperature_mh_sample
from LOTlib.Inference.TemperedTransitions import tempered_transitions_sample
from LOTlib.Inference.ParallelTempering import parallel_tempering_sample
from SimpleMPI.MPI_map import MPI_map
from SimpleMPI.ParallelBufferedIO import ParallelBufferedIO

from copy import copy
import sys

# We need to do this so that we can load via pickle (it searches for Shared)
sys.path.append("/home/piantado/Desktop/mit/Libraries/LOTlib/LOTlib/Examples/Number/") 

DATA_SIZE = 300

TEST_SAMPLES = 10000 # 10000
RUNS = 1000

TARGET_FILE = "/home/piantado/Desktop/mit/Libraries/LOTlib/LOTlib/Examples/Number/mpirun-Dec2013.pkl" # load a small file. The large one is only necessary if we want the "correct" target likelihood and top N numbers; if we just look at Z we don't need it!
DATA_FILE = "data/evaluation-data.pkl"
output = ParallelBufferedIO("evaluation.txt") # Use buffered output, and do it before we load the models (so that each  ParallelBufferedIO subprocess doesn't load)

# FOR DEBUGGING:
#TARGET_FILE = "tmp-hypotheses.pkl"
#DATA_FILE = "data/evaluation-data.pkl"
#output = sys.stdout

data   = pickle_load(DATA_FILE)

# recompute the target posterior in case it's diff data than was generated
# target here must be a dict from hypotheses to posteriors
target = dict()
for h in pickle_load(TARGET_FILE).get_all():
	sum(h.compute_posterior(data)) # add up the components returned by compute_posterior
	
	# Throw out any (if we changed max depth or anything)
	if h.lp > -Infinity: target[h] = h.lp

## A wrapper function for MPI_map
def run_one(r):
	if LOTlib.SIG_INTERRUPTED: return

	h0 = NumberExpression(G)
	
	#sampler = tempered_transitions_sample(copy(h0), data, TEST_SAMPLES, skip=0, temperatures=[1.0, 1.25, 1.5])
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="TemperedTransitions-1.5\t"+str(r), output=output )
	
	#sampler = tempered_transitions_sample(copy(h0), data, TEST_SAMPLES, skip=0, temperatures=[1.0, 1.05, 1.1])
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="TemperedTransitions-1.1\t"+str(r), output=output )
	
	#sampler = tempered_transitions_sample(copy(h0), data, TEST_SAMPLES, skip=0, temperatures=[1.0, 1.025, 1.05])
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="TemperedTransitions-1.05\t"+str(r), output=output )
	
	
	
	#sampler = parallel_tempering_sample(copy(h0), data, TEST_SAMPLES, within_steps=10, temperatures=[1.0, 1.25, 1.5], swaps=1)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="ParallelTempering-1.5\t"+str(r), output=output )
	
	#sampler = parallel_tempering_sample(copy(h0), data, TEST_SAMPLES, within_steps=10, temperatures=[1.0, 1.05, 1.1], swaps=1)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="ParallelTempering-1.1\t"+str(r), output=output )
	
	#sampler = parallel_tempering_sample(copy(h0), data, TEST_SAMPLES, within_steps=10, temperatures=[1.0, 1.025, 1.05], swaps=1)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="ParallelTempering-1.05\t"+str(r), output=output )
	
	
	
	sampler = datawise_optimize(copy(h0), data, TEST_SAMPLES, inner_steps=25, data_weight=1.0)
	evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="DatawiseOptimize-1.0\t"+str(r), output=output )
	
	sampler = datawise_optimize(copy(h0), data, TEST_SAMPLES, inner_steps=25, data_weight=0.1)
	evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="DatawiseOptimize-0.1\t"+str(r), output=output )

	sampler = datawise_optimize(copy(h0), data, TEST_SAMPLES, inner_steps=25, data_weight=0.01)
	evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="DatawiseOptimize-0.01\t"+str(r), output=output )
	
	
	#sampler = ptaboo_search( copy(h0), data, steps=TEST_SAMPLES, skip=0, seen_penalty=1.0)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="PtabooSearch-1.0\t"+str(r), trace=False, output=output )

	#sampler = ptaboo_search( copy(h0), data, steps=TEST_SAMPLES, skip=0, seen_penalty=10.0)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="PtabooSearch-10.0\t"+str(r), trace=False, output=output )

	#sampler = ptaboo_search( copy(h0), data, steps=TEST_SAMPLES, skip=0, seen_penalty=100.0)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="PtabooSearch-100.0\t"+str(r), trace=False, output=output )



	#sampler = increase_temperature_mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0, increase_amount=1.01)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="IncreaseTemperature-1.01\t"+str(r), trace=False, output=output)

	#sampler = increase_temperature_mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0, increase_amount=1.1)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="IncreaseTemperature-1.1\t"+str(r), trace=False, output=output)
	
	#sampler = increase_temperature_mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0, increase_amount=1.5)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="IncreaseTemperature-1.5\t"+str(r), trace=False, output=output)
	
	#sampler = increase_temperature_mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0, increase_amount=2.0)
	#evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="IncreaseTemperature-2.0\t"+str(r), trace=False, output=output)

	
	
	sampler = mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0)
	evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="BasicSampler\t"+str(r), trace=False, output=output)
	
	sampler = mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0, temperature=1.01)
	evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="BasicSampler-T1.01\t"+str(r), trace=False, output=output)
	
	sampler = mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0, temperature=1.05)
	evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="BasicSampler-T1.05\t"+str(r), trace=False, output=output )
	
	sampler = mh_sample( copy(h0), data, steps=TEST_SAMPLES, skip=0, temperature=1.1)
	evaluate_sampler(target, sampler, steps=TEST_SAMPLES, name="BasicSampler-T1.1\t"+str(r), trace=False, output=output )
	

# Actually run, in parallel!
MPI_map( run_one, range(RUNS) ) 


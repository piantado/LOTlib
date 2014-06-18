# -*- coding: utf-8 -*-


"""
	Evaluate one vs many chains
	
	Can run on mpi cluster as:
	
	mpiexec -n 8 python Number_evaluate_samplers.py
"""

from LOTlib.Examples.Number.Shared import *
from LOTlib.Evaluation.Evaluation import evaluate_sampler
from LOTlib.Inference.MetropolisHastings import mh_sample as the_sampler
from LOTlib.Miscellaneous import * # particularly weave
from SimpleMPI.MPI_map import MPI_map
from SimpleMPI.ParallelBufferedIO import ParallelBufferedIO

from copy import copy
import sys

# We need to do this so that we can load via pickle (it searches for Shared)
sys.path.append("/home/piantado/Desktop/mit/Libraries/LOTlib/LOTlib/Examples/Number/") 

DATA_SIZE = 300

TEST_SAMPLES = 10000 # 10000
RUNS = 1000

TARGET_FILE = "/home/piantado/Desktop/mit/Libraries/LOTlib/LOTlib/Examples/Number/mpirun-Dec2013.pkl" 
DATA_FILE = "data/evaluation-data.pkl"
output = ParallelBufferedIO("evaluation-chains.txt") # Use buffered output, and do it before we load the models (so that each  ParallelBufferedIO subprocess doesn't load). NOTE: Be sure to close it!

# FOR DEBUGGING:
#TARGET_FILE = "tmp-hypotheses.pkl"
#DATA_FILE = "data/evaluation-data.pkl"
#output = sys.stdout

data   = pickle_load(DATA_FILE)

# recompute the target posterior in case it's diff data than was generated
# target here must be a dict from hypotheses to posteriors
target = { h: sum(h.compute_posterior(data)) for h in pickle_load(TARGET_FILE).get_all()}

## A wrapper function for MPI_map
def run_one(r):
	if LOTlib.SIG_INTERRUPTED: return

	h0 = NumberExpression(grammar)
	
	for chain_count in [1,5, 10, 50, 100, 500, 1000]:
		if LOTlib.SIG_INTERRUPTED: return
		
		# make all of the chains using weave -- really we could run each for shorter, TEST_SAMPLES/chain_count
		g = weave( *[the_sampler( copy(h0), data, steps=TEST_SAMPLES, skip=0) for ci in xrange(chain_count) ] )
		
		evaluate_sampler(target, g, steps=TEST_SAMPLES, print_every=1000, name="Chain-"+str(chain_count), output=output)
		

# Actually run, in parallel!
MPI_map( run_one, range(RUNS) ) 

output.close() #Must do this or subprocesses hang around!

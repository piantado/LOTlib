# -*- coding: utf-8 -*-

"""
	All out rational-rules style gibbs on lexicons.
	For MPI or local.
	This is much slower than the vectorized versions.
	
	MPI run:
	$ mpiexec --hostfile ../../hosts.mpich2 -n 15 python2.7-mpi Search_MCMC.py
"""

from Shared import *

CHAINS = 2  #how many times do we run?
DATA_AMOUNTS = [800] # range(0,1500,100)
SAMPLES = 1000000
TOP_COUNT = 50
OUT_PATH = "/home/piantado/Desktop/mit/Libraries/LOTlib/examples/QuantifierLexicon/data/mcmc-run.pkl"

QUIET = False
RUN_MPI = False # should we run on MPI

########################################################################
## MPI imports if we need them
if RUN_MPI:
	from SimpleMPI.MPI_map import MPI_map

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# the main sampling function to run

#one run with these parameters
def run(data_size):
	
	print "Running ", data_size
	
	# We store the top 100 from each run
	hypset = FiniteBestSet(TOP_COUNT, max=True) 
	
	# initialize the data
	data = generate_data(data_size)
	
	# starting hypothesis -- here this generates at random
	learner = GriceanSimpleLexicon(G, args=['A', 'B', 'S'])
	
	# initialize all the words in learner
	for w in target.all_words():
		learner.set_word(w, G.generate('START')) # each word returns a true, false, or undef (None)
	
	# populate the finite sample by running the sampler for this many steps
	for x in LOTlib.MetropolisHastings.mh_sample(learner, data, SAMPLES, skip=0):
		print x
		hypset.push(x, x.lp)
	
	if RUN_MPI: print rank, "is done ", data_size
	
	return hypset
	
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MPI interface

# choose the appropriate map function
if RUN_MPI:
	allret = MPI_map(run, map(lambda x: [x], DATA_AMOUNTS * CHAINS)) # this many chains
else:
	allret = map(run,  DATA_AMOUNTS * CHAINS)

## combine into a single hypothesis set and save
outhyp = FiniteBestSet(max=True)
for r in allret: 
	print "# Merging ", len(r)
	outhyp.merge(r)
outhyp.save(OUT_PATH)

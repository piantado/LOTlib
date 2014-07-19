# -*- coding: utf-8 -*-
"""
	Simple evaluation of number schemes -- read LOTlib.Performance.Evaluation to see what the output is
"""

from LOTlib import lot_iter
from Shared import *
import os
from itertools import product


from optparse import OptionParser

parser = OptionParser()
parser.add_option("--out", dest="OUT", type="string", help="Output prefix", default="./evaluation-inference")
parser.add_option("--samples", dest="SAMPLES", type="int", default=100000, help="Number of samples to run")
parser.add_option("--ndata", dest="NDATA", type="int", default=300, help="Number of data points")
options, _ = parser.parse_args()

# Load the data
data = generate_data(options.NDATA)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MPI run mh_sample
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# These get defined for each process
from SimpleMPI.ParallelBufferedIO import ParallelBufferedIO
from LOTlib.Performance.Evaluation import evaluate_sampler


from LOTlib.Inference.TemperedTransitions import tempered_transitions_sample
from LOTlib.Inference.ParallelTempering import parallel_tempering_sample
from LOTlib.Inference.MetropolisHastings import mh_sample
from LOTlib.Inference.TabooMCMC import TabooMCMC
from LOTlib.Inference.ParticleSwarm import ParticleSwarm
from LOTlib.Inference.MultipleChainMCMC import MultipleChainMCMC

# Where we store the output
out_hypotheses = open(os.devnull,'w') #ParallelBufferedIO(options.OUT + "-hypotheses.txt")
out_aggregate  = ParallelBufferedIO(options.OUT + "-aggregate.txt")

def run_one(prefix, s):
	# prefix, sampler string
	
	# make a new hypothesis
	h0 = NumberExpression(grammar)
	generate_h0 = lambda: NumberExpression(grammar)

	# Create a sampler
	if s == 'mh_sample_A':               sampler = mh_sample(h0, data, options.SAMPLES,  likelihood_temperature=1.0)
	elif s == 'mh_sample_B':             sampler = mh_sample(h0, data, options.SAMPLES,  likelihood_temperature=1.1)
	elif s == 'mh_sample_C':             sampler = mh_sample(h0, data, options.SAMPLES,  likelihood_temperature=1.25)
	elif s == 'mh_sample_D':             sampler = mh_sample(h0, data, options.SAMPLES,  likelihood_temperature=2.0 )
	elif s == 'mh_sample_E':             sampler = mh_sample(h0, data, options.SAMPLES,  likelihood_temperature=5.0 )
	elif s == 'particle_swarm_A':        sampler = ParticleSwarm(generate_h0, data, steps=options.SAMPLES, within_steps=10)
	elif s == 'particle_swarm_B':        sampler = ParticleSwarm(generate_h0, data, steps=options.SAMPLES, within_steps=100)
	elif s == 'particle_swarm_C':        sampler = ParticleSwarm(generate_h0, data, steps=options.SAMPLES, within_steps=200)
	elif s == 'multiple_chains_A':       sampler = MultipleChainMCMC(generate_h0, data, steps=options.SAMPLES, chains=10)
	elif s == 'multiple_chains_B':       sampler = MultipleChainMCMC(generate_h0, data, steps=options.SAMPLES, chains=100)
	elif s == 'multiple_chains_C':       sampler = MultipleChainMCMC(generate_h0, data, steps=options.SAMPLES, chains=1000)
	elif s == 'parallel_tempering_A':    sampler = parallel_tempering_sample(h0, data, options.SAMPLES, within_steps=10, temperatures=[1.0, 1.025, 1.05], swaps=1)
	elif s == 'parallel_tempering_B':    sampler = parallel_tempering_sample(h0, data, options.SAMPLES, within_steps=10, temperatures=[1.0, 1.05, 1.1], swaps=1)
	elif s == 'parallel_tempering_C':    sampler = parallel_tempering_sample(h0, data, options.SAMPLES, within_steps=10, temperatures=[1.0, 1.25, 1.5], swaps=1)
	elif s == 'taboo_A':                 sampler = TabooMCMC( h0, data, steps=options.SAMPLES, skip=0, penalty=.10)
	elif s == 'taboo_B':                 sampler = TabooMCMC( h0, data, steps=options.SAMPLES, skip=0, penalty=1.0)
	elif s == 'taboo_C':                 sampler = TabooMCMC( h0, data, steps=options.SAMPLES, skip=0, penalty=10.0)
	else: assert False, s
	
	# Run evaluate on it, printing to the right locations
	evaluate_sampler(sampler, prefix="\t".join(map(str, [prefix, s])),  out_hypotheses=out_hypotheses, out_aggregate=out_aggregate)
 

# For each process, create the lsit of parameter
#params = [ list(g) for g in product(range(100), [1.0], [0.1, 0.2, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 5.0, 10.0, 100.0] )]

params = [ list(g) for g in product(range(100), ['multiple_chains_A', 'multiple_chains_B', 'multiple_chains_C', 
												'taboo_A', 'taboo_B', 'taboo_C', 
												'particle_swarm_A', 'particle_swarm_B', 'particle_swarm_C', 
												'mh_sample_A', 'mh_sample_B', 'mh_sample_C', 'mh_sample_D', 'mh_sample_E',
												 'parallel_tempering_A', 'parallel_tempering_B', 'parallel_tempering_C',
												])]
"""								
"""
from SimpleMPI.MPI_map import MPI_map

MPI_map(run_one, params, random_order=False)

out_hypotheses.close()
out_aggregate.close()


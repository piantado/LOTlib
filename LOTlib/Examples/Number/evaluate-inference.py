# -*- coding: utf-8 -*-
"""
	Simple evaluation of number schemes -- read LOTlib.Performance.Evaluation to see what the output is
"""

from Shared import *
import os
from itertools import product


from optparse import OptionParser

parser = OptionParser()
parser.add_option("--out", dest="OUT", type="string", help="Output prefix", default="./evaluation-inference")
parser.add_option("--samples", dest="SAMPLES", type="int", default=10000, help="Number of samples to run")
parser.add_option("--ndata", dest="NDATA", type="int", default=100, help="Number of data points")
options, _ = parser.parse_args()

# Load the data
data = generate_data(options.NDATA)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MPI run mh_sample
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# These get defined for each process
from SimpleMPI.ParallelBufferedIO import ParallelBufferedIO
from LOTlib.Performance.Evaluation import evaluate_sampler


from LOTlib.Inference.IncreaseTemperatureMH import increase_temperature_mh_sample
from LOTlib.Inference.ParallelTempering import parallel_tempering_sample
from LOTlib.Inference.MetropolisHastings import mh_sample
from LOTlib.Inference.ProbTaboo import ptaboo_search

# Where we store the output
out_hypotheses = open(os.devnull,'w') # ParallelBufferedIO(options.OUT + "-hypotheses.txt")
out_aggregate  = ParallelBufferedIO(options.OUT + "-aggregate.txt")

def run_one(prefix, s):
	# prefix, sampler string
	
	# make a new hypothesis
	h0 = NumberExpression(grammar)

	# Create a sampler
	if s == 'mh_sample_A':               sampler = mh_sample(h0, data, options.SAMPLES,  likelihood_temperature=1.0)
	elif s == 'mh_sample_B':             sampler = mh_sample(h0, data, options.SAMPLES,  likelihood_temperature=2.0, )
	elif s == 'mh_sample_C':             sampler = mh_sample(h0, data, options.SAMPLES,  likelihood_temperature=10.0 )
	elif s == 'parallel_tempering_A':    sampler = parallel_tempering_sample(h0, data, options.SAMPLES, within_steps=10, temperatures=[1.0, 1.025, 1.05], swaps=1)
	elif s == 'parallel_tempering_B':    sampler = parallel_tempering_sample(h0, data, options.SAMPLES, within_steps=10, temperatures=[1.0, 1.05, 1.1], swaps=1)
	elif s == 'parallel_tempering_C':    sampler = parallel_tempering_sample(h0, data, options.SAMPLES, within_steps=10, temperatures=[1.0, 1.25, 1.5], swaps=1)
	elif s == 'increase_temperature_A':  sampler = increase_temperature_mh_sample( h0, data, steps=options.SAMPLES, skip=0, increase_amount=1.0)
	elif s == 'increase_temperature_B':  sampler = increase_temperature_mh_sample( h0, data, steps=options.SAMPLES, skip=0, increase_amount=1.05)
	elif s == 'increase_temperature_C':  sampler = increase_temperature_mh_sample( h0, data, steps=options.SAMPLES, skip=0, increase_amount=1.1)
	elif s == 'ptaboo_A':                sampler = ptaboo_search( h0, data, steps=options.SAMPLES, skip=0, seen_penalty=1.0)
	elif s == 'ptaboo_B':                sampler = ptaboo_search( h0, data, steps=options.SAMPLES, skip=0, seen_penalty=10.0)
	elif s == 'ptaboo_C':                sampler = ptaboo_search( h0, data, steps=options.SAMPLES, skip=0, seen_penalty=10.0)
	else: assert False, s
	
	# Run evaluate on it, printing to the right locations
	evaluate_sampler(sampler, prefix="\t".join(map(str, [prefix, s])),  out_hypotheses=out_hypotheses, out_aggregate=out_aggregate)
 

# For each process, create the lsit of parameter
#params = [ list(g) for g in product(range(100), [1.0], [0.1, 0.2, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 5.0, 10.0, 100.0] )]

params = [ list(g) for g in product(range(100), ['mh_sample_A', 'mh_sample_B', 'mh_sample_C', 
											   'parallel_tempering_A', 'parallel_tempering_B', 'parallel_tempering_C',
											   'increase_temperature_A', 'increase_temperature_B', 'increase_temperature_C',
											   'ptaboo_A', 'ptaboo_B', 'ptaboo_C']) ]

from SimpleMPI.MPI_map import MPI_map

MPI_map(run_one, params)

out_hypotheses.close()
out_aggregate.close()

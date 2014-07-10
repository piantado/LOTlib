# -*- coding: utf-8 -*-
"""
	Simple evaluation of number schemes
"""

from LOTlib import lot_iter
from Shared import *
import os
from itertools import product


from optparse import OptionParser

parser = OptionParser()
parser.add_option("--out", dest="OUT", type="string", help="Output prefix", default="./evaluation-temperatures")
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
from LOTlib.Inference.TemperedTransitions import tempered_transitions_sample
from LOTlib.Inference.ParallelTempering import parallel_tempering_sample
from LOTlib.Inference.MetropolisHastings import mh_sample
from LOTlib.Inference.ProbTaboo import ptaboo_search

# Where we store the output
out_hypotheses = open(os.devnull,'w') #ParallelBufferedIO(options.OUT + "-hypotheses.txt")
out_aggregate  = ParallelBufferedIO(options.OUT + "-aggregate.txt")

def run_one(prefix, pt, lt):
	# prefix, prior_temperature and likelihood_temperature
	
	# make a new hypothesis
	h0 = NumberExpression(grammar)

	sampler = mh_sample(h0, data, options.SAMPLES,  likelihood_temperature=lt, prior_temperature=pt)
	
	# Run evaluate on it, printing to the right locations
	evaluate_sampler(sampler, prefix="\t".join(map(str, [prefix, pt, lt])),  out_hypotheses=out_hypotheses, out_aggregate=out_aggregate)
 

# For each process, create the list of parameter
params = [ list(g) for g in product(range(100), [1.0], [0.1, 0.2, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 5.0, 10.0, 100.0] )]

from SimpleMPI.MPI_map import MPI_map

MPI_map(run_one, params)

out_hypotheses.close()
out_aggregate.close()


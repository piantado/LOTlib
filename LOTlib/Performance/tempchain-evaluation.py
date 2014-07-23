# -*- coding: utf-8 -*-
"""
	Simple evaluation of number schemes
"""

from LOTlib import lot_iter
import os
from itertools import product

from SimpleMPI.MPI_map import MPI_map, synchronize_variable
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--out", dest="OUT", type="string", help="Output prefix", default="output/inference")
parser.add_option("--samples", dest="SAMPLES", type="int", default=100000, help="Number of samples to run")
parser.add_option("--repetitions", dest="REPETITONS", type="int", default=100, help="Number of repetitions to run")
parser.add_option("--model", dest="MODEL", type="str", default="Number", help="Which model to run on (Number, Galileo, RationalRules, SimpleMagnetism)")
parser.add_option("--print-every", dest="PRINTEVERY", type="int", default=1000, help="Evaluation prints every this many")
options, _ = parser.parse_args()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the test model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if options.MODEL == "Number":
	# Load the data
	from LOTlib.Examples.Number.Shared import generate_data, NumberExpression, grammar 
	data = synchronize_variable( lambda : generate_data(300)  )
	generate_h0 = lambda: NumberExpression(grammar)

elif options.MODEL == "Galileo":
	from LOTlib.Hypotheses.GaussianLOTHypothesis import GaussianLOTHypothesis
	from LOTlib.Examples.SymbolicRegression.Galileo import data, grammar
	generate_h0 = lambda:  GaussianLOTHypothesis(grammar)

elif options.MODEL == "RationalRules":
	
	from LOTlib.Examples.RationalRules.Shared import grammar, data
	from LOTlib.Hypotheses.RationalRulesLOTHypothesis import RationalRulesLOTHypothesis
	from LOTlib.DefaultGrammars import DNF
	generate_h0 = lambda: RationalRulesLOTHypothesis(grammar=DNF)
	
elif options.MODEL == "SimpleMagnetism":
	from LOTlib.Examples.Magnetism.SimpleMagnetism import grammar, data
	from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
	generate_h0 = lambda: LOTHypothesis(grammar, args=['x', 'y'], ALPHA=0.999)

else:
	assert false, "Unimplemented model: %s" % options.MODEL

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MPI run mh_sample
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# These get defined for each process
from SimpleMPI.ParallelBufferedIO import ParallelBufferedIO
from LOTlib.Performance.Evaluation import evaluate_sampler
from LOTlib.Inference.MultipleChainMCMC import MultipleChainMCMC

# Where we store the output
out_hypotheses = open(os.devnull,'w') #ParallelBufferedIO(options.OUT + "-hypotheses.txt")
out_aggregate  = ParallelBufferedIO(options.OUT + "-aggregate.txt")

def run_one(iteration, nchains, temperature):

	sampler = MultipleChainMCMC(generate_h0, data, steps=options.SAMPLES, chains=nchains, likelihood_temperature=temperature)
	
	# Run evaluate on it, printing to the right locations
	evaluate_sampler(sampler, prefix="\t".join(map(str, [options.MODEL, iteration, nchains, temperature])),  out_hypotheses=out_hypotheses, out_aggregate=out_aggregate)
 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create all parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# For each process, create the list of parameter
params = map(list, product( range(100), [1,2,5,10,50,100,500,1000], [0.1, 0.2, 0.5, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 5.0] ))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Actually run
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MPI_map(run_one, params, random_order=False)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Clean up
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

out_hypotheses.close()
out_aggregate.close()


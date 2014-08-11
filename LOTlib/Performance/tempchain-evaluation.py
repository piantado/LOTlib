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
parser.add_option("--out", dest="OUT", type="string", help="Output prefix", default="output/tempchain")
parser.add_option("--samples", dest="SAMPLES", type="int", default=100000, help="Number of samples to run")
parser.add_option("--repetitions", dest="REPETITONS", type="int", default=100, help="Number of repetitions to run")
parser.add_option("--model", dest="MODEL", type="str", default="Number", help="Which model to run on (Number, Galileo, RationalRules, SimpleMagnetism)")
parser.add_option("--print-every", dest="PRINTEVERY", type="int", default=1000, help="Evaluation prints every this many")
options, _ = parser.parse_args()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the test model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if options.MODEL == "Number":
    from LOTlib.Examples.Number.Shared import generate_data, grammar,  make_h0
    data = synchronize_variable( lambda : generate_data(300)  )
elif options.MODEL == "Galileo":
    from LOTlib.Examples.SymbolicRegression.Galileo import data, grammar, make_h0
elif options.MODEL == "RationalRules":
    from LOTlib.Examples.RationalRules.Shared import grammar, data, make_h0
elif options.MODEL == "SimpleMagnetism":
    from LOTlib.Examples.Magnetism.SimpleMagnetism import grammar, data, make_h0
elif options.MODEL == "RegularExpression":
    from LOTlib.Examples.RegularExpression.RegularExpressionInferece import grammar, data, make_h0
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

    sampler = MultipleChainMCMC(make_h0, data, steps=options.SAMPLES, nchains=nchains, likelihood_temperature=temperature)

    # Run evaluate on it, printing to the right locations
    evaluate_sampler(sampler, prefix="\t".join(map(str, [options.MODEL, iteration, nchains, temperature])),  out_hypotheses=out_hypotheses, out_aggregate=out_aggregate)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create all parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# For each process, create the list of parameter
params = map(list, product( range(options.REPETITONS), [1,10,100], [0.1, 0.5, 1.0, 2.0, 10.0] ))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Actually run
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MPI_map(run_one, params, random_order=False)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Clean up
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

out_hypotheses.close()
out_aggregate.close()

MPI_done()

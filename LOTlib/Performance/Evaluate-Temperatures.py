# -*- coding: utf-8 -*-
"""
    Evaluate at different temperatures
"""
from itertools import product
from optparse import OptionParser

from Evaluation import load_model
from LOTlib.MPI.MPI_map import MPI_map, get_rank

parser = OptionParser()
parser.add_option("--out", dest="OUT", type="string", help="Output prefix", default="output/tempchain")
parser.add_option("--samples", dest="SAMPLES", type="int", default=100000, help="Number of samples to run")
parser.add_option("--repetitions", dest="REPETITONS", type="int", default=100, help="Number of repetitions to run")
parser.add_option("--model", dest="MODEL", type="str", default="Number", help="Which model to run on (Number, Galileo, RationalRules, SimpleMagnetism)")
parser.add_option("--print-every", dest="PRINTEVERY", type="int", default=1000, help="Evaluation prints every this many")
options, _ = parser.parse_args()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# These get defined for each process
from LOTlib.Performance.Evaluation import evaluate_sampler
from LOTlib.Inference.MultipleChainMCMC import MultipleChainMCMC

def run_one(iteration, chains, temperature):

    data, make_h0 = load_model(options.MODEL)

    h0 = make_h0()
    sampler = MultipleChainMCMC(make_h0, data, steps=options.SAMPLES, nchains=chains, likelihood_temperature=temperature)

    with open(options.OUT+"/aggregate.%s" % get_rank(), 'a') as out_aggregate:
        with open(options.OUT+"/hypotheses.%s" % get_rank(),'a')  as out_hypotheses:
            # Run evaluate on it, printing to the right locations
            evaluate_sampler(sampler, trace=False, prefix="\t".join(map(str, [options.MODEL, iteration, chains, temperature])),
                             out_hypotheses=out_hypotheses, out_aggregate=out_aggregate, print_every=options.PRINTEVERY)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create all parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# For each process, create the list of parameter
params = map(list, product( range(options.REPETITONS), [1,10,100], [0.1, 0.5, 1.0, 2.0, 10.0] ))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Actually run
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MPI_map(run_one, params, random_order=False)


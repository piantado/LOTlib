# -*- coding: utf-8 -*-
"""
	Simple evaluation of proposal schemes. Test various mixtures of them. 
"""

import os
from itertools import product
from optparse import OptionParser

from SimpleMPI.MPI_map import MPI_map, synchronize_variable
from LOTlib.Miscellaneous import q


parser = OptionParser()
parser.add_option("--out", dest="OUT", type="string", help="Output prefix", default="output/proposal")
parser.add_option("--samples", dest="SAMPLES", type="int", default=100000, help="Number of samples to run")
parser.add_option("--chains", dest="CHAINS", type="int", default=10, help="Number of chains to run in parallel")
parser.add_option("--repetitions", dest="REPETITONS", type="int", default=100, help="Number of repetitions to run")
parser.add_option("--model", dest="MODEL", type="str", default="Number100", help="Which model to run on (Number, Galileo, RationalRules, SimpleMagnetism)")
parser.add_option("--print-every", dest="PRINTEVERY", type="int", default=1000, help="Evaluation prints every this many")
options, _ = parser.parse_args()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the test model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if options.MODEL == "Number100":
    # Load the data
    from LOTlib.Examples.Number.Global import generate_data, grammar,  make_h0
    data = synchronize_variable( lambda : generate_data(100)  )

elif options.MODEL == "Number300":
    # Load the data
    from LOTlib.Examples.Number.Global import generate_data, grammar,  make_h0
    data = synchronize_variable( lambda : generate_data(300)  )
    
elif options.MODEL == "Number1000":
    # Load the data
    from LOTlib.Examples.Number.Global import generate_data, grammar,  make_h0
    data = synchronize_variable( lambda : generate_data(1000)  )

elif options.MODEL == "Galileo":
	from LOTlib.Examples.SymbolicRegression.Galileo import data, grammar, make_h0	
elif options.MODEL == "RationalRules":
	from LOTlib.Examples.RationalRules.Global import grammar, data, make_h0
elif options.MODEL == "SimpleMagnetism":
	from LOTlib.Examples.Magnetism.Simple.Global import grammar, data, make_h0
elif options.MODEL == "RegularExpression":
	from LOTlib.Examples.RegularExpression.Global import grammar, data, make_h0
else:
	assert false, "Unimplemented model: %s" % options.MODEL

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MPI run mh_sample
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# These get defined for each process
from SimpleMPI.ParallelBufferedIO          import ParallelBufferedIO
from LOTlib.Performance.Evaluation         import evaluate_sampler
from LOTlib.Inference.MultipleChainMCMC    import MultipleChainMCMC
from LOTlib.Inference.Proposals.MixtureProposal      import MixtureProposal
from LOTlib.Inference.Proposals.RegenerationProposal import RegenerationProposal
from LOTlib.Inference.Proposals.InsertDeleteProposal import InsertDeleteProposal
from LOTlib.Inference.Proposals.InverseInlineProposal import InverseInlineProposal

# Where we store the output
out_hypotheses = open(os.devnull,'w') #ParallelBufferedIO(options.OUT + "-hypotheses.txt")
out_aggregate  = ParallelBufferedIO(options.OUT + "-aggregate.txt")

def run_one(iteration, probs=None):

	m = MixtureProposal([RegenerationProposal(grammar), InsertDeleteProposal(grammar), InverseInlineProposal(grammar)], probs=probs )
		
	# define a wrapper to set this proposal
	def wrapped_make_h0():
		h0 = make_h0()
		h0.set_proposal_function(m)
		return h0

	sampler = MultipleChainMCMC(wrapped_make_h0, data, steps=options.SAMPLES, nchains=options.CHAINS)
	
	# Run evaluate on it, printing to the right locations
	evaluate_sampler(sampler, prefix="\t".join(map(str, [options.MODEL, iteration, q(str(probs)) ])),  out_hypotheses=out_hypotheses, out_aggregate=out_aggregate)
 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create all parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# For each process, create the list of parameter
params = map(list, product( range(options.REPETITONS), [ (1., 1., 1.), 
                                                         (2., 1., 1.),
                                                         (1., 2., 1.),
                                                         (1., 1., 2.),
                                                         (2., 2., 1.),
                                                         (1., 2., 2.),
                                                         (2., 1., 2.),
                                                         (10., 1., 1.),
                                                         (1., 10., 1.),
                                                         (1., 1., 10.),
                                                         (10., 10., 1.),
                                                         (1., 10., 10.),
                                                         (10., 1., 10.)
                                                         ]))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Actually run
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MPI_map(run_one, params, random_order=False)



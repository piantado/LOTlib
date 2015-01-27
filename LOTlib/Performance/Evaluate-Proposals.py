# -*- coding: utf-8 -*-
"""
    Simple evaluation of proposal schemes. Test various mixtures of them.
"""

from itertools import product
from optparse import OptionParser

from Evaluation import load_model
from LOTlib.MPI.MPI_map import MPI_map, get_rank
from LOTlib.Miscellaneous import q

parser = OptionParser()
parser.add_option("--out", dest="OUT", type="string", help="Output prefix", default="output-Proposals")
parser.add_option("--samples", dest="SAMPLES", type="int", default=100000, help="Number of samples to run")
parser.add_option("--chains", dest="CHAINS", type="int", default=10, help="Number of chains to run in parallel")
parser.add_option("--repetitions", dest="REPETITONS", type="int", default=100, help="Number of repetitions to run")
parser.add_option("--print-every", dest="PRINTEVERY", type="int", default=1000, help="Evaluation prints every this many")
options, _ = parser.parse_args()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MPI run mh_sample
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# These get defined for each process
from LOTlib.Performance.Evaluation         import evaluate_sampler
from LOTlib.Inference.MultipleChainMCMC    import MultipleChainMCMC
from LOTlib.Inference.Proposals.MixtureProposal      import MixtureProposal
from LOTlib.Inference.Proposals.RegenerationProposal import RegenerationProposal
from LOTlib.Inference.Proposals.InsertDeleteProposal import InsertDeleteProposal
from LOTlib.Inference.Proposals.InverseInlineProposal import InverseInlineProposal

def run_one(iteration, model, probs=None):

    data, make_h0 = load_model(model)
    htmp = make_h0() # just use this to get the grammar
    grammar = htmp.grammar

    m = MixtureProposal([RegenerationProposal(grammar), InsertDeleteProposal(grammar), InverseInlineProposal(grammar)], probs=probs )

    # define a wrapper to set this proposal
    def wrapped_make_h0():
        h0 = make_h0()
        h0.set_proposal_function(m)
        return h0

    sampler = MultipleChainMCMC(wrapped_make_h0, data, steps=options.SAMPLES, nchains=options.CHAINS)

    with open(options.OUT+"/aggregate.%s" % get_rank(), 'a') as out_aggregate:
        with open(options.OUT+"/hypotheses.%s" % get_rank(),'a')  as out_hypotheses:
            evaluate_sampler(sampler, trace=False, prefix="\t".join(map(str, [model, iteration, q(str(probs)) ])),
                             out_hypotheses=out_hypotheses, out_aggregate=out_aggregate, print_every=options.PRINTEVERY)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create all parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# For each process, create the list of parameter
params = map(list, product( range(options.REPETITONS),
                            ['SymbolicRegression.Galileo', 'Magnetism.Simple', 'RationalRules', 'RegularExpression', 'Number:100', 'Number:300', 'Number:1000'],
                            [ (1., 1., 1.),
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



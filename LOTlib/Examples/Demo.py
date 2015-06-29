# -*- coding: utf-8 -*-
"""
A quick demo for running any example. This gives a "standard sampling" algorithm that can be used to
watch an MCMC trace from the examples.

NOTE: ctrl-c defaultly will break
"""

import LOTlib
import pickle
from LOTlib.FunctionNode import cleanFunctionNodeString
from LOTlib.MCMCSummary.TopN import TopN
from LOTlib.Miscellaneous import qq, display_option_summary
from LOTlib import break_ctrlc

from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler

def standard_sample(make_hypothesis, make_data, skip=9, show=True, N=100, save_top='top.pkl', alsoprint='None', **kwargs):
    """
        Just a simplified interface for sampling, allowing printing (showing), returning the top, and saving.
        This is used by many examples, and is meant to easily allow running with a variety of parameters.
        NOTE: This skip is a skip *only* on printing
        **kwargs get passed to sampler
    """
    if LOTlib.SIG_INTERRUPTED:
        return TopN()  # So we don't waste time!

    h0 = make_hypothesis()
    data = make_data()

    best_hypotheses = TopN(N=N)

    f = eval(alsoprint)

    for i, h in enumerate(break_ctrlc(MHSampler(h0, data, **kwargs))):
        best_hypotheses.add(h)

        if show and i%(skip+1) == 0:
            print i, \
                h.posterior_score, \
                h.prior, \
                h.likelihood, \
                f(h) if f is not None else '', \
                qq(cleanFunctionNodeString(h))

    if save_top is not None:
        print "# Saving top hypotheses"
        with open(save_top, 'w') as f:
            pickle.dump(best_hypotheses, f)

    return best_hypotheses



if __name__ == "__main__":

    # ========================================================================================================
    # Process command line arguments
    # ========================================================================================================

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--model", dest="MODEL", type="string", default="Number",
                      help="Which model do we run? (e.g. 'Number', 'Magnetism.Simple', etc.")
    parser.add_option("--alsoprint", dest="ALSO_PRINT", type="string", default="None",
                      help="A function of a hypothesis we can also print at the start of a line to see things we "
                           "want. E.g. --alsoprint='lambda h: h.get_knower_pattern()' ")
    (options, args) = parser.parse_args()

    display_option_summary(options)

    # ========================================================================================================
    # Load the model specified on the command line
    # ========================================================================================================

    from LOTlib.Examples.ExampleLoader import load_example

    make_hypothesis, make_data = load_example(options.MODEL)

    # ========================================================================================================
    #  Run the example's standard sampler with these parameters
    # ========================================================================================================

    # This is just a wrapper that nicely prints information
    standard_sample(make_hypothesis, make_data, alsoprint=options.ALSO_PRINT)


# -*- coding: utf-8 -*-
"""
A quick demo for running any example. This gives a "standard sampling" algorithm that can be used to
watch an MCMC trace from the examples.

NOTE: ctrl-c defaultly will break
"""

if __name__ == "__main__":
    from LOTlib.Miscellaneous import Infinity

    # ========================================================================================================
    # Process command line arguments
    # ========================================================================================================

    from optparse import OptionParser
    parser = OptionParser()

    parser.add_option("--model", dest="MODEL", type="string", default="Number",
                      help="Which model do we run? (e.g. 'Number', 'Magnetism.Simple', etc.")
    parser.add_option("--steps", dest="STEPS", type="int", default=Infinity, help="Draw this many samples")
    parser.add_option("--skip", dest="SKIP", type="int", default=0, help="Skip this many steps between samples")
    parser.add_option("--alsoprint", dest="ALSO_PRINT", type="string", default="None",
                      help="A function of a hypothesis we can also print at the start of a line to see things we "
                           "want. E.g. --alsoprint='lambda h: h.get_knower_pattern()' ")
    (options, args) = parser.parse_args()

    from LOTlib.Miscellaneous import display_option_summary

    display_option_summary(options)

    # ========================================================================================================
    # Load the model specified on the command line
    # ========================================================================================================

    from LOTlib.Examples import load_example

    make_hypothesis, make_data = load_example(options.MODEL)

    # ========================================================================================================
    #  Run the example's standard sampler with these parameters
    # ========================================================================================================

    from LOTlib.Inference.Samplers.StandardSample import standard_sample

    # This is just a wrapper that nicely prints information
    standard_sample(make_hypothesis, make_data, alsoprint=options.ALSO_PRINT, steps=options.STEPS, skip=options.SKIP)


"""
Unit test to make sure examples each load and can make a hypothesis and data, and support sampling
"""

import unittest

from LOTlib.Examples import load_example
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler

class ExampleLoaderTest(unittest.TestCase):

    def runTest(self):

        for model in ['EvenOdd', 'FOL', 'Magnetism.Simple', 'Magnetism.Complex',
                      'NAND', 'Number', 'RegularExpression', 'RationalRules',
                      'StochasticGrammarInduction', 'SymbolicRegression.Galileo',
                      'SymbolicRegression.Symbolic', 'Prolog', 'PureLambda', 'Lua']:
            print "# Testing loading of example", model

            make_hypothesis, make_data = load_example(model)

            d  = make_data()
            d  = make_data(10) # require an amount

            # Let's just try initializing a bunch of times
            for _ in xrange(100):
                h0 = make_hypothesis()

            # and ensure that the samplign will run
            for _ in MHSampler(h0, d, steps=100):
                pass

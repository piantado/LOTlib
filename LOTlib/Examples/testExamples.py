"""
Unit test to make sure examples each load and can make a hypothesis and data, and support sampling
"""

import unittest

from LOTlib.Examples import load_example
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler

class ExampleLoaderTest(unittest.TestCase):

    def runTest(self):

        for model in ['EvenOdd', 'FOL', 'Magnetism.Simple', 'Magnetism.Complex',
                      'NAND', 'Number', 'RegularExpression', 'RationalRules.TwoConcepts',
                      'StochasticGrammarInduction', 'SymbolicRegression.Galileo',
                      'SymbolicRegression.Symbolic']:
            print "# Testing loading of example ", model

            make_hypothesis, make_data = load_example(model)

            h0 = make_hypothesis()
            d  = make_data()
            d  = make_data(10) # require an amount

            for _ in MHSampler(h0, d, steps=10):
                pass

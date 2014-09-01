"""
class to test Proposals.py
follows the standards in https://docs.python.org/2/library/unittest.html
"""

from LOTlib import lot_iter
import unittest

from LOTlib.Inference.Proposals import *
from Grammars import FunctionNodeGrammar
from collections import defaultdict

class ProposalsTest(unittest.TestCase):

    # initialization that happens before each test is carried out
    def setUp(self):
        self.G = FunctionNodeGrammar.g
        
        pass
    

    # function that is executed after each test is carried out
    def tearDown(self):
        pass




# A Test Suite composed of all tests in this class
def suite():
    return unittest.TestLoader().loadTestsFromTestCase(ProposalsTest)


if __name__ == '__main__':
    unittest.main()

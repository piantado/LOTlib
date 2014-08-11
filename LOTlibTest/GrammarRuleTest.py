"""
class to test GrammarRule.py
follows the standards in https://docs.python.org/2/library/unittest.html
"""

import unittest

from LOTlib.GrammarRule import *
class GrammarRuleTest(unittest.TestCase):

    # initialization that happens before each test is carried out
    def setUp(self):
        pass





    # function that is executed after each test is carried out
    def tearDown(self):
        pass


# A Test Suite composed of all tests in this class
def suite():
    return unittest.TestLoader().loadTestsFromTestCase(GrammarRuleTest)




if __name__ == '__main__':
    unittest.main()

"""
class to test Proposals.py
follows the standards in https://docs.python.org/2/library/unittest.html
"""

import unittest

from LOTlib.Proposals import *
class ProposalsTest(unittest.TestCase):
	
	# initialization that happens before each test is carried out
	def setUp(self):
		pass
	
	
	
	
	
	# function that is executed after each test is carried out
	def tearDown(self):
		pass
	



# A Test Suite composed of all tests in this class
def suite():
	return unittest.TestLoader().loadTestsFromTestCase(ProposalsTest)


if __name__ == '__main__':
	unittest.main()





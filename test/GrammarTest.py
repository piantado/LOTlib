"""
	class to test Grammar.py
	follows the standards in https://docs.python.org/2/library/unittest.html
"""


import unittest

from LOTlib.Grammar import *
import math


class GrammarTest(unittest.TestCase):



	# initialization that happens before each test is carried out
	def setUp(self):
		self.G = Grammar()
		self.G.add_rule('START', 'A ', ['START'], 0.1)
		self.G.add_rule('START', 'B ', ['START'], 0.3)
		self.G.add_rule('START', 'NULL', [], 0.6)
	
	# tests .log_probability() function
	def test_log_probability(self):
		# sample from G 100 times
		for i in range(100):
			t = self.G.generate('START')
			s = t.__str__()
			# test whether the object is a string (this should be a test for the pystring function in FunctionNode.py)
			# self.assertEqual(type(t), str)
			# count probability manually
			prob = self.countProbability(s)
			# check that it's equal to .log_probability()
			self.assertTrue(prob - t.log_probability() < 0.00000001)
			# print s, t.log_probability()


	# counts the probability of the grammar manually
	# NOTE: not modular at this point, if we change our test grammar this function will return something incorrect
	# NOTE: also only works if START -> any characters not in NULL (fix)
	def countProbability(self, string):
		# count number of occurrences of A and B
		a = string.count('A')
		b = string.count('B')
		return math.log(0.1**a * 0.3**b * 0.6)













# main code to run the test
if __name__ == '__main__':
	unittest.main()





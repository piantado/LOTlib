"""
class to test FunctionNode.py
follows the standards in https://docs.python.org/2/library/unittest.html
"""

import unittest

# from LOTlib.FunctionNode import * # THIS THROWS AN ERROR "ImportError: cannot import name isFunctionNode". Is there an import loop?
from LOTlib.Grammar import *


class FunctionNodeTest(unittest.TestCase):
	
	# initialization that happens before each test is carried out
	def setUp(self):
		self.G = Grammar()
		self.G.add_rule('START', 'A ', ['START'], 0.1)
		self.G.add_rule('START', 'B ', ['START'], 0.3)
		self.G.add_rule('START', 'NULL', [], 0.6)
	
	# tests the .pystring() method
	def str(self):
		t = self.G.generate('START')
		string = t.pystring()
		# test whether the object is a string
		self.assertEqual(type(string), str)
		self.assertEqual(type(t.__str__()), str)
	
	
	
	# function that is executed after each test is carried out
	def tearDown(self):
		pass
	






if __name__ == '__main__':
	unittest.main()





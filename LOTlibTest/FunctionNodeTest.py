"""
class to test FunctionNode.py
follows the standards in https://docs.python.org/2/library/unittest.html
"""

import unittest
from LOTlib.FunctionNode import *
from LOTlib.Grammar import *
from LOTlibTest.Grammars import FunctionNodeGrammar
from LOTlib import lot_iter


class FunctionNodeTest(unittest.TestCase):
	
	# initialization that happens before each test is carried out
	def setUp(self):
		self.G = FunctionNodeGrammar.g

	
	# tests the .pystring() method
	def test_str(self):
		t = self.G.generate('START')
		string = t.pystring()
		# test whether the object is a string
		self.assertEqual(type(string), str)
		self.assertEqual(type(t.__str__()), str)
	
	def test_eq(self):
		counter = 0
		for i in lot_iter(xrange(10000)):
			x = self.G.generate()
			y = self.G.generate()

			if x.pystring() == y.pystring():
				counter += 1
				# print(counter)
				# print(x.pystring()+'\n'+y.pystring()+'\n')

			self.assertEqual(x.pystring() == y.pystring(), x == y, "Without bvs, the pystrings should be the same")

	
	# function that is executed after each test is carried out
	def tearDown(self):
		pass


# A Test Suite composed of all tests in this class
def suite():
	return unittest.TestLoader().loadTestsFromTestCase(FunctionNodeTest)
	
if __name__ == '__main__':
	unittest.main()

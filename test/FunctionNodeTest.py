"""
class to test FunctionNode.py
follows the standards in https://docs.python.org/2/library/unittest.html
"""

import unittest
from LOTlib.FunctionNode import *
from LOTlib.Grammar import *


class FunctionNodeTest(unittest.TestCase):
	
	# initialization that happens before each test is carried out
	def setUp(self):
		self.G = Grammar()
		self.G.add_rule('START', 'S', ['NP', 'VP'], 0.1)
		self.G.add_rule('NP', 'NV', ['DET', 'N'], 0.6)
		self.G.add_rule('NP', 'NV', ['DET', 'ADJ', 'N'], 0.4)
		self.G.add_rule('NP', 'NV', ['PN'], 0.3)
		self.G.add_rule('NP', 'lambda', ['N'], 0.5,  bv_type="N", bv_args=None)
		self.G.add_rule('VP', 'NV', ['V', 'NP'], 0.5)
		self.G.add_rule('N', 'ball', None, 0.2)
		self.G.add_rule('N', 'computer', None, 0.2)
		self.G.add_rule('N', 'phone', None, 0.2)
		self.G.add_rule('PN', 'Chomsky', None, 0.3)
		self.G.add_rule('PN', 'Samay', None, 0.3)
		self.G.add_rule('PN', 'Steve', None, 0.3)
		self.G.add_rule('PN', 'Hassler', None, 0.3)
		self.G.add_rule('V', 'eats', None, 0.25)
		self.G.add_rule('V', 'kills', None, 0.25)
		self.G.add_rule('V', 'maims', None, 0.25)
		self.G.add_rule('V', 'sees', None, 0.25)
		self.G.add_rule('ADJ', 'peculiar', None, 0.4)
		self.G.add_rule('ADJ', 'strange', None, 0.4)
		self.G.add_rule('ADJ', 'red', None, 0.4)
		self.G.add_rule('ADJ', 'queasy', None, 0.4)
		self.G.add_rule('ADJ', 'happy', None, 0.4)
		self.G.add_rule('DET', 'the', None, 0.5)
		self.G.add_rule('DET', 'a', None, 0.5)

	
	# tests the .pystring() method
	def test_str(self):
		t = self.G.generate('START')
		string = t.pystring()
		# test whether the object is a string
		self.assertEqual(type(string), str)
		self.assertEqual(type(t.__str__()), str)
	
	def test_eq(self):
		counter = 0
		for i in xrange(100000):	
			x = self.G.generate()
			y = self.G.generate()

			if x.pystring() == y.pystring():
				counter += 1
				print(counter)
				print(x.pystring()+'\n'+y.pystring()+'\n')

			self.assertEqual(x.pystring() == y.pystring(), x == y, "Without bvs, the pystrings should be the same")

	
	# function that is executed after each test is carried out
	def tearDown(self):
		pass
	
if __name__ == '__main__':
	unittest.main()

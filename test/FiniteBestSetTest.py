"""
	class to test FiniteBestSet.py
	follows the standards in https://docs.python.org/2/library/unittest.html
"""


import unittest

from LOTlib.FiniteBestSet import *
import random

class FiniteBestSetTest(unittest.TestCase):



	# initialization that happens before each test is carried out
	def setUp(self):
		self.ar = range(100)
		random.shuffle(self.ar)
		self.max_set = [90,91,92,93,94,95,96,97,98,99]
		self.min_set = [0,1,2,3,4,5,6,7,8,9]

	# tests that the 10 best elements in the priority queue are indeed 90-99
	def test_max(self):
		Q = FiniteBestSet(N=10)
		for x in self.ar: Q.add(x,x)
		self.assertTrue(set(Q.get_all()).issuperset( set(self.max_set)))
		self.assertTrue(set(Q.get_all()).issubset( set(self.max_set)))


	# tests that the 10 best elements in the priority queue are indeed 0-9
	# (for a priority queue that favors low numbers)
	def test_min(self):
		Q = FiniteBestSet(N=10, max=False)
		for x in self.ar: Q.add(x,x)
		self.assertTrue(set(Q.get_all()).issuperset( set(self.min_set)))
		self.assertTrue(set(Q.get_all()).issubset( set(self.min_set)))


	# tests that the queue contains the correct elements (as returned by the __contains__ method)
	def test_contains(self):
		Q = FiniteBestSet(N=10)
		for x in self.ar: Q.add(x,x)
		for elem in self.max_set:
			self.assertTrue(Q.__contains__(elem))
		for elem in self.min_set:
			self.assertFalse(Q.__contains__(elem))

	# tests that the queue contains the correct amount of elements after each step (as returned by the __len__ method)
	def test_len(self):
		Q = FiniteBestSet(N=10)
		count = 0
		self.assertEqual(Q.__len__(), 0)
		for x in self.ar:
			Q.add(x,x)
			count = min(count+1, 10)
			self.assertEqual(Q.__len__(), count)

	###################### testing the add function ######################

	# tests that the assertion for checking p==None in iterators is functioning

	# tests that an assertion error is raised if we didn't supply any key to the element

	# tests that we don't add an element that's already in the queue

	# miscellaneous




# main code to run the test
if __name__ == '__main__':
	unittest.main()






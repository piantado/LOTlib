"""
class to test Memoization.py
follows the standards in https://docs.python.org/2/library/unittest.html
"""

import unittest

from LOTlib.Memoization import *


class MemoizationTest(unittest.TestCase):
	
	# initialization that happens before each test is carried out
	def setUp(self):
		self.d = BoundedDictionary(N=100)

	# test BoundedDictionary __setitem__
	def test__setitem__(self):
		# add elements to the BoundedDictionary
		for i in range(300):
			self.d.__setitem__(i, 2*i)
			# make sure self.d contains i
			self.d.__contains__(i)
			# make sure self.d contains the correct number of items (should empty the dictionary after 201 adds)
			self.assertEqual(self.d.dict_size, (i+1)%201)
			# make sure self.d contains the correct last counts
			if (i+1)%201 != 0:
				self.assertEqual(self.d.last_count[i], 0)

	# test BoundedDictionary __getitem__
	def test__getitem__(self):
		# add elements to the BoundedDictionary
		for i in range(300):
			self.d.__setitem__(i, 2*i)
			# get another item
			if i >= 50:
				self.assertEqual(self.d.__getitem__(i-50), 2*(i-50))
				# make sure self.d contains the correct number of items
				self.assertEqual(self.d.dict_size, min(200, i+1))
				# make sure last_count's are correct
				self.assertEqual(self.d.last_count[i], i-50)
	
	
	
	
	
	# function that is executed after each test is carried out
	def tearDown(self):
		self.assertTrue(self.d.dict_size <= 200)
	






if __name__ == '__main__':
	unittest.main()





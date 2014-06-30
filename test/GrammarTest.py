"""
	class to test Grammar.py
	follows the standards in https://docs.python.org/2/library/unittest.html
"""


import unittest

from LOTlib.Grammar import *
import math
from collections import defaultdict


class GrammarTest(unittest.TestCase):



	# initialization that happens before each test is carried out
	def setUp(self):
		self.G = Grammar()
		self.G.add_rule('START', 'A ', ['START'], 0.1)
		self.G.add_rule('START', 'B ', ['START'], 0.3)
		self.G.add_rule('START', 'NULL', None, 0.6)
	
	# tests .log_probability() function
	def test_log_probability(self):
		# sample from G 100 times
		for i in range(100):
			t = self.G.generate('START')
			# count probability manually
			prob = self.countProbability(t)
			# check that it's equal to .log_probability()
			self.assertTrue(prob - t.log_probability() < 0.00000001)


	# tests repeated sampling
	def test_sampling(self):
		# sample from G 10,000 times and record the frequency in a dictionary
		frequencyDictionary = defaultdict(lambda: 0)
		for i in range(10000):
			t = self.G.generate('START')
			frequencyDictionary[t] += 1
		# compare the log probabilities with the sampling
		for tree in frequencyDictionary:
			logProb = tree.log_probability()
			print log(frequencyDictionary[tree]/10000.), logProb
			# TODO: come up with/look up a good metric for converting the similarities between counts and log probabilities
			# into a number that gives us a measure of "how good the sampling went". For this simple grammar, it looks pretty good.

	# counts the probability of the grammar manually
	# NOTE: not modular at this point, if we change our test grammar this function will return something incorrect
	# NOTE: also only works if START -> any characters not in NULL (fix)
	def countProbability(self, node):
		# count number of occurrences of A and B
		ls = node.as_list()
		# recursively go through the tree, counting up the number of a's and b's
		counts = self.count(ls)
		# print ls, counts
		return math.log(0.1**counts['A '] * 0.3**counts['B '] * 0.6)


	# counts the number of occurrences of each element in a nested list of strings
	# returns a dictionary
	def count(self, ls):
		# http://stackoverflow.com/questions/9358983/python-dictionary-and-default-values
		dictionary = defaultdict(lambda: 0)
		# recursively flatten the list
		flattenedList = self.flatten(ls)
		# count the elements one-by-one
		for elem in flattenedList:
			dictionary[elem] += 1
		# return the dictionary
		return dictionary

	# flattens a nested list
	def flatten(self, ls):
		newlist = []
		for elem in ls:
			if type(elem) == list:
				newlist.extend(self.flatten(elem))
			else: newlist.append(elem)
		return newlist














# main code to run the test
if __name__ == '__main__':
	unittest.main()

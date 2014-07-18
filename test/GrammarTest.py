"""
	class to test Grammar.py
	follows the standards in https://docs.python.org/2/library/unittest.html
"""


import unittest

from LOTlib.Grammar import *
from LOTlib.Proposals import RegenerationProposal
import math
from collections import defaultdict


class GrammarTest(unittest.TestCase):


	# initialization that happens before each test is carried out
	def setUp(self):
		# self.G = Grammar()
		# # NOTE: these probabilities should get normalized
		# self.G.add_rule('START', 'S', ['NP', 'VP'], 0.1)
		# self.G.add_rule('START', 'S', ['INTERJECTION'], 0.3)
		# self.G.add_rule('NP', 'NV', ['DET', 'N'], 0.6)
		# self.G.add_rule('NP', 'NV', ['DET', 'ADJ', 'N'], 0.4)
		# self.G.add_rule('NP', 'NV', ['PN'], 0.3)
		# self.G.add_rule('VP', 'NV', ['V', 'NP'], 0.5)
		# self.G.add_rule('N', 'ball', None, 0.2)
		# self.G.add_rule('N', 'computer', None, 0.2)
		# self.G.add_rule('N', 'phone', None, 0.2)
		# self.G.add_rule('PN', 'Chomsky', None, 0.3)
		# self.G.add_rule('PN', 'Samay', None, 0.3)
		# self.G.add_rule('PN', 'Steve', None, 0.3)
		# self.G.add_rule('PN', 'Hassler', None, 0.3)
		# self.G.add_rule('V', 'eats', None, 0.25)
		# self.G.add_rule('V', 'kills', None, 0.25)
		# self.G.add_rule('V', 'maims', None, 0.25)
		# self.G.add_rule('V', 'sees', None, 0.25)
		# self.G.add_rule('ADJ', 'peculiar', None, 0.4)
		# self.G.add_rule('ADJ', 'strange', None, 0.4)
		# self.G.add_rule('ADJ', 'red', None, 0.4)
		# self.G.add_rule('ADJ', 'queasy', None, 0.4)
		# self.G.add_rule('ADJ', 'happy', None, 0.4)
		# self.G.add_rule('DET', 'the', None, 0.5)
		# self.G.add_rule('DET', 'a', None, 0.5)
		# self.G.add_rule('INTERJECTION', 'sh*t', None, 0.6)
		# self.G.add_rule('INTERJECTION', 'fu*k pi', None, 0.6)
		self.G = Grammar()
		self.G.add_rule('START', 'A ', ['START'], 0.1)
		self.G.add_rule('START', 'B ', ['START'], 0.3)
		self.G.add_rule('START', 'NULL', None, 0.6)


	# tests .lp_regenerate_propose_to function
	def test_lp_regenerate_propose_to(self):
		# run the test 100 times
		for i in range(1):
			# generate two trees X and Y
			X = self.G.generate('START')
			Y = self.G.generate('START')
			# the number of times tree regenerated from X matches tree Y
			num = 0
			# the RegenerationProposal class
			rp = RegenerationProposal(self.G)
			# call propose_tree on x 1000 times
			for j in range(1000):
				tree = rp.propose_tree(X)[0]
				print 'trees are: ', X, tree, Y
				# see if the tree matches Y
				if Y.__eq__(tree): num += 1
			print num
			# TODO: check with a chi-square test
			# (but chi-square tests can only be done with finite grammars) (as they are over a finite distribution)

	# tests that the generation and regeneration of trees is consistent with the probabilities
	# that are output by lp_regenerate_propose_to
	def test_generation(self):
		# Enumerate the top 20 trees based on their probability of being generated
		# put each of the top 20 trees in its own category, and the rest in another category
		categories = self.get_top_trees(20) # this should return an array of 20 trees
		expected_counts = self.get_expected_counts(categories) # this should return a dictionary of 21 expected "counts"
		actual_counts = {tree: 0 for tree in categories}
		actual_counts[None] = 0 # for the 21st category
		# Generate 1000 trees at random
		trees = []
		for i in range(1000):
			tree = self.G.generate('START')
			trees.append(tree)
			if tree in categories:
				actual_counts[tree] += 1
			else:
				actual_counts[None] += 1
		# see if the frequency with which each category of trees is generated matches the
		# expected counts using a chi-squared test
		chi_squared_statistic, p = self.get_chi_squared_statistic(expected_counts, actual_counts)
		# if p > 0.01, test passes
		self.assertTrue(p > 0.01, "Trees are not being generated according to the expected log probabilities")
		# If this test passes, for each 1000 trees:
		p_values = []
		for tree in trees:
			# Enumerate the top 20 trees that could be generated and put them in separate categories (with the rest of the trees in another category)
			tree_categories = self.get_top_trees(20, tree=tree)
			tree_expected_counts = self.get_expected_counts(tree_categories, tree=tree)
			tree_actual_counts = {newtree: 0 for newtree in tree_categories}
			tree_actual_counts[None] = 0
			# Regenerate 1000 trees at random
			newtrees = []
			for i in range(1000):
				newtree = rp.propose_tree(tree)[0]
				newtrees.append(newtree)
				if newtree in tree_categories:
					tree_actual_counts[newtree] += 1
				else:
					tree_actual_counts[None] += 1
			# see if the frequency with which each category of trees is generated matches
			# the expected counts using a chi-squared test (generate a p-value)
			css, pvalue = self.get_chi_squared_statistic(tree_expected_counts, tree_actual_counts)
			p_values.append(pvalue)
		# See if these 1000 p-values follow a uniform distribution by generating another "master" p-value
		master_p_value = self.get_uniform_statistic(p_values)
		# if p > 0.01, test passes
		self.assertTrue(master_p_value > 0.01, "Trees are not being regenerated according to their expected log probabilities")
	
	# FUNCTIONS TO IMPLEMENT:
	def get_top_trees(self, num, tree=None):
		# TODO: implement generating one layer of a tree in Grammar (i.e. function that only replaces one node)
		pass

	def get_expected_counts(self, tree_categories, tree=None):
		pass

	def get_chi_squared_statistic(self, expected_counts, actual_counts):
		pass

	def get_uniform_statistic(self, p_values):
		pass

	# tests .log_probability() function
	def test_log_probability(self):
		# construct a different grammar
		# self.G = Grammar()
		# self.G.add_rule('START', 'A ', ['START'], 0.1)
		# self.G.add_rule('START', 'B ', ['START'], 0.3)
		# self.G.add_rule('START', 'NULL', None, 0.6)

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
			# print log(frequencyDictionary[tree]/10000.), logProb
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

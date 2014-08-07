"""
	A small finite grammar with bound variables that don't have any args.
"""


from LOTlib.Grammar import Grammar
import math

g = Grammar()
g.add_rule("START", "S", ["NP", "VP"], 1.0)
g.add_rule("NP", "NP", ["det", "N"], 1.0)
g.add_rule("VP", "VP", ["V", "NP"], 1.0)
g.add_rule("VP", "VP", ["is", "AGE", "years old"], 1.0)
g.add_rule("det", "the", None, 1.0)
g.add_rule("N", "cat", None, 1.0)
g.add_rule("N", "human", None, 1.0)
g.add_rule("V", "likes", None, 1.0)
g.add_rule("V", "kills", None, 1.0)
g.add_rule("V", "eats", None, 1.0)
g.add_rule('AGE', '', ['INT'], 1.0,  bv_type='INT', bv_args=None, bv_prefix='a')
g.add_rule('INT', '20', None, 1.0)
g.add_rule('INT', '22', None, 1.0)

def log_probability(tree):
	ls = tree.as_list()
	# the probability is just the probability of the given noun phrase being generated, times the probability of the given verb phrase being generated
	prob_nounphrase = 0.5
	# thus, check if we introduced a bound variable or not
	if type(ls[2][1]) is list:
		prob_verbphrase = 1.0/12
	else: prob_verbphrase = 1.0/6
	return math.log(prob_nounphrase * prob_verbphrase)

if __name__ == "__main__":
	for i in xrange(100):
		tree = g.generate()
		print tree, tree.log_probability()

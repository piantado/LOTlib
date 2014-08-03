"""
	A simple infinite grammar with skewed probabilities
"""

from LOTlib.Grammar import Grammar


g = Grammar()
g.add_rule('START', 'A ', ['START'], 0.1)
g.add_rule('START', 'B ', ['START'], 0.3)
g.add_rule('START', 'NULL', None, 0.6)

def log_probability(tree):
	return 0 # TODO: stub

if __name__ == "__main__":
	for i in xrange(100):
		print(g.generate())

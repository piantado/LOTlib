"""
	A small finite grammar with bound variables that have bv_args.
"""


from LOTlib.Grammar import Grammar
import math

g = Grammar()

g.add_rule("START", 'S', "A", 1.0, bv_type="A", bv_args=["B"], bv_prefix="p")
g.add_rule("A", 'a', "B", 1.0)
g.add_rule("A", 'a', "w", 1.0)
g.add_rule("A", 'a', "x", 1.0)
g.add_rule("B", 'b', "y", 1.0)
g.add_rule("B", 'b', "z", 1.0)

def log_probability(tree):
	ls = tree.as_list()
	# if the first argument to the list is not 'p0', then we don't have to deal with the bound variable
	if ls[1][0] == 'a' and type(ls[1][1]) is not list:
		return math.log(0.25)
	elif ls[1][0] == 'a':
		return math.log(0.25*0.5)
	# otherwise, we have a bound variable
	else:
		return math.log(0.25*0.5)

if __name__ == "__main__":
	for i in xrange(100):
		print(g.generate())

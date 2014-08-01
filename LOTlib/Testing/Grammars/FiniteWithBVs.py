"""
	A small finite grammar with bound variables that don't have any args.
"""

from LOTlib.Grammar import Grammar

g = Grammar()

g.add_rule("START", '', "A", 1.0, bv_type="P", bv_args=["B,B"], bv_prefix="p")
g.add_rule("START", '', "A", 1.0, bv_type="P", bv_args=["A,B"], bv_prefix="p")
g.add_rule("A", 'a', "B", 1.0)
g.add_rule("A", 'a', "w", 1.0)
g.add_rule("A", 'a', "x", 1.0)
g.add_rule("B", 'b', "y", 1.0)
g.add_rule("B", 'b', "z", 1.0)

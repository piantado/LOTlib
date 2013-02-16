
"""
	This class is a wrapper for representing "rules" in the grammar. 
"""

from math import log
from LOTlib.Miscellaneous import assert_or_die

class GrammarRule:
	def __init__(self, nt, name, to, rid, p=1.0, resample_p=1.0, bv=[]):
		"""	
			nt - the nonterminal
			name - the name of this function
			to - what you expand to (usually a FunctionNode). 
			rid - the rule id number
			p - unnormalized probability of expansion
			resample_p - in resampling, what is the probability of choosing this node?		
			bv - what bound variables were introduced?
			
			A rule where "to" is a nonempty list is a real expansion:
				GrammarRule( "EXPR", "plus", ["EXPR", "EXPR"], ...) -> plus(EXPR,EXPR)
			A rule where "to" is [None] is a thunk 
				GrammarRule( "EXPR", "plus", [None], ...) -> plus()
			A rule where "to" is [] is a real terminal (non-thunk)
				GrammarRule( "EXPR", "five", [], ...) -> five
			A rule where "name" is '' expands without parens:
				GrammarRule( "EXPR", '', "SUBEXPR", ...) -> EXPR->SUBEXPR				
						
			NOTE: The rid is very important -- it's what we use to determine equality
		"""
		self.__dict__.update(locals())
		self.lp = log(p)
		
		if name == '': assert_or_die( len(to) == 1, "GrammarRules with empty names must have only 1 argument")
		
		
	def __repr__(self):
		return str(self.nt) + " -> " + self.name + str(self.to) + "    [p=" +str(self.p)+ "; resample_p=" + str(self.resample_p) +"]"

	def __eq__(self, other): return ( self.rid == other.rid and self.nt == other.nt)
	def __ne__(self, other): return not self.__eq__(other)

class FunctionRule(GrammarRule):
	""" Just a subtype for when we want to pass distributions to values"""
	def __init__(self, nt, function, rid, p=1.0, resample_p=1.0):
		self.__dict__.update(locals())
		self.lp = log(p)
		
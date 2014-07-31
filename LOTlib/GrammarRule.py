
"""
	This class is a wrapper for representing "rules" in the grammar.
"""
# TODO: One day we will change "nt" to returntype to match with the FunctionNodes


class GrammarRule:
	def __init__(self, nt, name, to, rid, p=1.0, resample_p=1.0, bv_type=None, bv_args=None, bv_prefix="y", bv_p=None):
		"""
			*nt* - the nonterminal

			*name* - the name of this function

			*to* - what you expand to (usually a FunctionNode).

			*rid* - the rule id number

			*p* - unnormalized probability of expansion

			*resample_p* - the probability of choosing this node in an expansion

			*bv_type* - return type of the introduced bound variable 

			*bv_args* - what are the args when we use a bv (None is terminals, else a type signature)

			Examples:
			A rule where "expansion" is a nonempty list is a real expansion:
				GrammarRule( "EXPR", "plus", ["EXPR", "EXPR"], ...) -> plus(EXPR,EXPR)
			A rule where "expansion" is [] is a thunk
				GrammarRule( "EXPR", "plus", [], ...) -> plus()
			A rule where "expansion" is [] is a real terminal (non-thunk)
				GrammarRule( "EXPR", "five", None, ...) -> five
			A rule where "name" is '' expands without parens:
				GrammarRule( "EXPR", '', "SUBEXPR", ...) -> EXPR->SUBEXPR

			NOTE: The rule id (rid) is very important -- it's what we use expansion determine equality
		"""
		p = float(p)  # make sure these are floats
		
		self.__dict__.update(locals())
		
		if name == '':

			assert (to is None) or (len(to) == 1), "*** GrammarRules with empty names must have only 1 argument"

	def __repr__(self):
		return str(self.nt) + " -> " + self.name + (str(self.to) if self.to is not None else '') + "\t\t[p=" +str(self.p)+ "; resample_p=" + str(self.resample_p) +"]" + "<BV:"+ str(self.bv_type)+";"+str(self.bv_args)+";"+self.bv_prefix+">"

	def __eq__(self, other):
		return ( self.rid == other.rid and self.nt == other.nt)

	def __ne__(self, other):
		return not self.__eq__(other)

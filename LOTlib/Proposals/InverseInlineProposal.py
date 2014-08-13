from LOTProposal import LOTProposal
from LOTlib.Miscellaneous import lambdaTrue
from copy import copy
from random import random
from math import log

from LOTlib.FunctionNode import FunctionNode


class InverseInlineProposal(LOTProposal):
	"""
		Inverse inlinling for non-functions
		
		TODO: Make this use the grammar instead of inventing function nodes and variable names from nowhere
		
		TODO: Probability of a is not computed correctly -- must sum over all equivalent as in computing the forward probability 
		
		TODO: Is-extractable is too stringent -- could allow bound variables defined in parents of ni
	"""
	
	def __init__(self, grammar):
		"""
			This takes a grammar and a regex to match variable names
		"""
		self.__dict__.update(locals())
		LOTProposal.__init__(self, grammar)
		
	def propose_tree(self, t):
		"""
			Delete:
				- find an apply
				- take the interior of the lambdathunk and sub it in for the lambdaarg everywhere, remove the apply
			Insert:
				- Find a node
				- Find a subnode s
				- Remove all repetitions of s, create a lambda thunk
				- and add an apply with the appropriate machinery
		"""

		newt = copy(t) 
		f,b = 0.0, 0.0
		success = False #acts to tell us if we found and replaced anything
			
		def is_extractable(n):
			# We must check that this doesn't contain any bound variables of outer lambdas
			introduced_bvs = set() # the bvs that are introduced below n (and are thus okay)
			for ni in n: 
				if ni.ruleid < 0 and ni.name not in introduced_bvs: # If it's a bv 
					return False
				elif ni.islambda() and ni.bv_name is not None:
					introduced_bvs.add(ni.bv_name)
			return True

		def is_apply(x):
				return (x.name == 'apply_') and (len(x.args)==2) and x.args[0].islambda() and not x.args[1].islambda()
	
		# ------------------
		if random() < 0.5: #INSERT MOVE
			
			# sample a node
			for ni, di, resample_p, Z in self.grammar.sample_node_via_iterate(newt):
				
				# Sample a subnode -- NOTE: we must use copy(ni) here since we modify this tree, and so all hell breaks loose otherwise
				for s, sdi, sresample_p, sZ in self.grammar.sample_node_via_iterate(copy(ni), predicate=is_extractable):
					success = True
					
					below = copy(ni)
					varname = 'Y'+str(di+1)
					
					# replace with the variables
					# TODO: FIX THE RID HERE -- HOW DO WE TREAT IT?
					below.replace_subnodes(s, FunctionNode(s.returntype, varname, None, ruleid=-999))
					
					# create a new node, the lambda abstraction
					fn = FunctionNode(below.returntype, 'apply_', [ \
						FunctionNode('LAMBDAARG', 'lambda', [ below ], bv_prefix='Y', bv_name=varname, bv_type=s.returntype, bv_args=[] ),\
						s
							] )
					
					# Now convert into a lambda abstraction
					ni.setto(fn) 
					
					f += (log(resample_p) - log(Z)) + (log(sresample_p) - log(sZ))  
			
			
		
		else: # DELETE MOVE
			
			resample_p = None
			for ni, di, resample_p, Z in self.grammar.sample_node_via_iterate(newt, predicate=is_apply):
				success = True
				
				## what does the variable look like? Here a thunk with bv_name
				var = FunctionNode( ni.args[0].bv_type , ni.args[0].bv_name, None)
				
				assert len(ni.args) == 2
				assert len(ni.args[0].args) == 1
				
				newni = ni.args[0].args[0] # may be able to optimize away?
				
				## and remove
				newni.replace_subnodes(var, ni.args[1])
				
				##print ":", newni
				ni.setto(newni) 
				f += (log(resample_p) - log(Z))
			
			if resample_p is None: return [newt,0.0]
			
			#newZ = self.grammar.resample_normalizer(newt, predicate=is_replacetype)
			##to go back, must choose the 
			#b += log(resample_p) - log(newZ)
		
	#	newt.fix_bound_variables() ## TODO: I think these are just from old versions
	#	newt.reset_function() # make sure we update the function
		if not success: 
			return [copy(t),0.0]
		else:
			return [newt, f-b]
	
if __name__ == "__main__":
		from LOTlib.Examples.Number.Shared import generate_data, grammar,  make_h0, NumberExpression
		
		p = InverseInlineProposal(grammar)
			
		for _ in xrange(100):	
			t = grammar.generate()
			print "\n\n", t
			for _ in xrange(10):
				t =  p.propose_tree(t)[0]
				print "\t", t
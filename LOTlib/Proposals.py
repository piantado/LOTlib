"""
	Proposal classes for LOTlib.
	
	These are really just functions that allow us to initialize and store some state (like a grammar)
	They all return [proposal, forwardp-backwardp] for use in MH.
	
	TODO: MixtureProposal only works correctly if both are ergodic. If not, we may need something special to do forward/backward
	
	NOTE TODO: INSERT/DELETE DOES NOT WORK -- MAY NOT MATCH CORRECTLY WITH BV_NAME NOW
"""

from LOTlib.Miscellaneous import weighted_sample, lambdaTrue, sample1
from LOTlib.FunctionNode import *
from copy import copy
from math import log
from random import random, randint
import numpy

class LOTProposal(object):
	"""
		A class of LOT proposals. This wraps calls with copying of the hypothesis
		so that we can implement only propose_t classes for subclasses, that generate trees
	"""
	def __init__(self, grammar):
		self.__dict__.update(locals())
		
	def __call__(self, h, **kwargs):
		# A wrapper that calls propose_tree (defined in subclasses) on our tree value
		# so this manages making LOTHypotheses (or the relevant subclass), and proposal subclasses
		# can just manage trees
		p = copy(h)
		ret = self.propose_tree(h.value, **kwargs) # don't unpack, since we may return [newt,fb] or [newt,f,b]
		p.set_value(ret[0])
		ret[0] = p
		return ret
		
		
class RegenerationProposal(LOTProposal):
	"""
		The default in LOTHypothesis
	"""
	
	def propose_tree(self, t, separate_fb=False, predicate=lambdaTrue):
		"""
			If separate_fb=True -- return [newt, f, b], instead of [newt,f-b]
		"""
		newt = copy(t)
		
		n, rp, tZ = None, None, None
		for ni, di, resample_p, Z in self.grammar.sample_node_via_iterate(newt, do_bv=True, predicate=predicate):
			n = ni
			
			# re-generate my type from the grammar, and change this functionode
			if self.grammar.is_nonterminal(n.returntype):
				n.setto(self.grammar.generate(n.returntype, d=di))
			else: pass # do nothing if we aren't returnable from the grammar
			
			tZ = Z
			
			rp = resample_p
		
		newZ = self.grammar.resample_normalizer(newt, predicate=predicate)
		
		#print "PROPOSED ", newt		
		f = (log(rp) - log(tZ))   + newt.log_probability()
		b = (log(rp) - log(newZ)) + t.log_probability()	
		
		if separate_fb:
			return [newt,f, b]
		else:
			return [newt,f-b]
	
class InsertDeleteProposal(LOTProposal):
	"""
		This class is a mixture of standard rejection proposals, and insert/delete proposals
		
		TODO: both insert and delete moves compute similar things, so maybe we can collapse them more elgantly?
			
		NOTE: Without these moves, you will often generate a useful part of a function in, say, an AND, and 
			you can't remove the AND, meaning you will just make it some subtree equal to "True" -- 
			e.g. this allows AND(T, True)  -> T, which is what you want. Otherwise, you have trouble moving out of those
		
		NOTE: This does not go on lambdas -- they're too hard to think about for now.. But even "not" doing them, there are asymmetries--we want to not treat them as "replicating rules", so we can't have sampled them, and also can't delete them
			
	"""
	
	def __init__(self, grammar, insert_delete_probability=0.5):
		self.__dict__.update(locals())
		
		# Must save this because we mix in RegenerationProposals
		self.my_regeneration_proposal = RegenerationProposal(grammar)
		
	def propose_tree(self, t):
		
		# Default regeneration proposal with some probability
		if random() >= self.insert_delete_probability: 
			return self.my_regeneration_proposal.propose_tree(t)
		
		newt = copy(t)
		fb = 0.0 # the forward/backward prob we return
		sampled=False # so we can see if we didn't do it
		
		if random() < 0.5: # So we insert
			
			# first sample a node (through sample_node_via_iterate, which handles everything well)
			for ni, di, resample_p, resample_Z in self.grammar.sample_node_via_iterate(newt):
				if ni.args is None: continue # Can't deal with these TODO: CHECK THIS?
				
				# Since it's an insert, see if there is a (replicating) rule that expands
				# from ni.returntype to some ni.returntype
				replicating_rules = filter(lambda x: x.name != 'lambda' and (x.to is not None) and any([a==ni.returntype for a in x.to]), self.grammar.rules[ni.returntype])
				
				# If there are none, then we can't insert!
				if len(replicating_rules) == 0: continue
				
				# choose a replicating rule; NOTE: this is done uniformly in this step, for simplicity
				r, gp = weighted_sample(replicating_rules, probs=lambda x: x.p, return_probability=True, log=False)
				gp = log(r.p) - sum([x.p for x in self.grammar.rules[ni.returntype]]) # this is the probability overall in the grammar, not my prob of sampling
				
				# Now take the rule and expand the children:
				
				# choose who gets to be ni
				nrhs = len( [ x for x in r.to if x == ni.returntype] ) # how many on the rhs are there?
				if nrhs == 0: continue
				replace_i = randint(0,nrhs-1) # choose the one to replace
				
				## Now expand args but only for the one we don't sample...
				args = []
				for x in r.to:
					if x == ni.returntype:
						if replace_i == 0: args.append( copy(ni) ) # if it's the one we replace into
						else:              args.append( self.grammar.generate(x, d=di+1) ) #else generate like normalized
						replace_i -= 1
					else:              
						args.append( self.grammar.generate(x, d=di+1) ) #else generate like normal	
							
				# Now we must count the multiple ways we could go forward or back
				after_same_children = [ x for x in args if x==ni] # how many are the same after?
				#backward_resample_p = sum([ x.resample_p for x in after_same_children]) # if you go back, you can choose any identical kids
				
				# create the new node
				sampled = True
				ni.setto( FunctionNode(returntype=r.nt, name=r.name, args=args, generation_probability=gp, bv_name=None, bv_args=None, ruleid=r.rid, resample_p=r.resample_p ) )
				
			if sampled:
				
				new_lp_below = sum(map(lambda z: z.log_probability(), filter(isFunctionNode, args))) - ni.log_probability()
				
				newZ = self.grammar.resample_normalizer(newt)
				# To sample forward: choose the node ni, choose the replicating rule, choose which "to" to expand (we could have put it on any of the replicating rules that are identical), and genreate the rest of the tree
				f = (log(resample_p) - log(resample_Z)) + -log(len(replicating_rules)) + (log(len(after_same_children))-log(nrhs)) + new_lp_below
				# To go backwards, choose the inserted rule, and any of the identical children, out of all replicators
				b = (log(ni.resample_p) - log(newZ)) + (log(len(after_same_children)) - log(nrhs))
				fb = f-b
				
		else: # A delete move!
			for ni, di, resample_p, resample_Z in self.grammar.sample_node_via_iterate(newt):
				if ni.name == 'lambda': continue # can't do anything
				if ni.args is None: continue # Can't deal with these TODO: CHECK THIS?
				
				# Figure out which of my children have the same type as me
				replicating_kid_indices = [ i for i in xrange(len(ni.args)) if isFunctionNode(ni.args[i]) and ni.args[i].returntype==ni.returntype]
				
				nrk = len(replicating_kid_indices) # how many replicating kids
				if nrk == 0: continue # if no replicating rules here
				
				## We need to compute a few things for the backwards probability
				replicating_rules = filter(lambda x: (x.to is not None) and any([a==ni.returntype for a in x.to]), self.grammar.rules[ni.returntype])
				if len(replicating_rules) == 0: continue
				
				i = sample1(replicating_kid_indices) # who to promote; NOTE: not done via any weighting
				
				# Now we must count the multiple ways we could go forward or back
				# Here, we could have sampled any of them equivalent to ni.args[i]
				
				before_same_children = [ x for x in ni.args if x==ni.args[i] ] # how many are the same after?
				
				# the lp of everything we'd have to create going backwards
				old_lp_below = sum(map(lambda z: z.log_probability(), filter(isFunctionNode, ni.args)  )) - ni.args[i].log_probability()
				
				# and replace it
				sampled = True
				ni.setto( copy(ni.args[i]) ) # TODO: copy not necessary here, I think?
				
			if sampled:
				
				newZ = self.grammar.resample_normalizer(newt)
				# To go forward, choose the node, and then from all equivalent children
				f = (log(resample_p) - log(resample_Z)) + (log(len(before_same_children)) - log(nrk))
				# To go back, choose the node, choose the replicating rule, choose where to put it, and generate the rest of the tree
				b = (log(ni.resample_p) - log(newZ))  + -log(len(replicating_rules)) + (log(len(before_same_children)) - log(nrk)) + old_lp_below
				fb = f-b
		
		# and fix the bound variables, whose depths may have changed
		if sampled: newt.fix_bound_variables()
		
		return [newt, fb]

class MixtureProposal(LOTProposal):
	"""
		A mixture of proposals, like
		
		m = MixtureProposal([RegenerationProposal(grammar), InsertDeleteProposal(grammar)] )
		
		Probabilities of each can be specified
	
	"""
	def __init__(self, proposals, probs=None):
		#print proposals
		self.__dict__.update(locals())
		
		if probs is None:
			self.probs = numpy.array( [1.] * len(proposals) )
		
	def propose_tree(self, t):
		p = weighted_sample(self.proposals, probs=self.probs, log=False)
		
		return p.propose_tree(t)

class InverseInlineProposal(LOTProposal):
	"""
		Inverse inlinling for non-functions
		
		TODO: HOW TO DEAL WITH ENSURING THE LAMBDA RULES ARE IN THE GRAMMAr, HAVE THE RIGHT PROBS, ETC
	
		TODO: NOT QUITE WORKING RIGHT -- 2014 JUL 25 -- BV NAMES ARE WRONG
		
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
						#FunctionNode('LAMBDATHUNK',  'lambda', [ s  ], bv_name=None, bv_type=None, bv_args=None)\
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
		
	#	newt.fix_bound_variables()
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
			

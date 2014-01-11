"""
	Proposal classes for LOTlib.
	
	These are really just functions that allow us to initialize and store some state (like a grammar)
	They all return [proposal, forwardp-backwardp] for use in MH.
"""

from LOTlib.Miscellaneous import weighted_sample
from LOTlib.FunctionNode import *
from copy import copy
from math import log
from random import random
import numpy

class LOTProposal(object):
	"""
		A class of LOT proposals. This wraps calls with copying of the hypothesis
		so that we can implement only propose_t classes for subclasses, that generate trees
	"""
	def __init__(self, grammar):
		self.__dict__.update(locals())
		
	def __call__(self, h):
		# A wrapper that calls propose_tree (defined in subclasses) on our tree value
		# so this manages making LOTHypotheses (or the relevant subclass), and proposal subclasses
		# can just manage trees
		p = copy(h)
		newt, fb = self.propose_tree(h.value)
		p.set_value(newt)
		return [p, fb]
		
		
class RegenerationProposal(LOTProposal):
	"""
		The default in LOTHypothesis
	"""
	
	def propose_tree(self, t):
		newt = copy(t)
		
		n, rp, tZ = None, None, None
		for ni, di, resample_p, Z in self.grammar.sample_node_via_iterate(newt, do_bv=True):
			n = ni
			
			# re-generate my type from the grammar, and change this functionode
			if self.grammar.is_nonterminal(n.returntype):
				n.setto(self.grammar.generate(n.returntype, d=di))
			else: pass # do nothing if we aren't returnable from the grammar
			
			tZ = Z
			
			rp = resample_p
		
		newZ = self.grammar.resample_normalizer(newt)
		
		#print "PROPOSED ", newt		
		f = (log(rp) - log(tZ))   + newt.log_probability()
		b = (log(rp) - log(newZ)) + t.log_probability()	
		
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
				r = sample1(replicating_rules)
				
				lp = r.lp - logsumexp([x.lp for x in self.grammar.rules[ni.returntype]]) # this is the probability overall in the grammar, not my prob of sampling
				
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
				ni.setto( FunctionNode(returntype=r.nt, name=r.name, args=args, lp=lp, bv_name=None, bv_args=None, ruleid=r.rid, resample_p=r.resample_p ) )
				
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
		
		m = MixtureProposal(grammar, [RegenerationProposal(grammar), InsertDeleteProposal(grammar)] )
		
		Probabilities of each can be specified
	
	"""
	def __init__(self, grammar, proposals, probs=None):
		#print proposals
		self.__dict__.update(locals())
		LOTProposal.__init__(self, grammar)
		
		if probs is None:
			self.probs = numpy.array( [1.] * len(proposals) )
		
	def propose_tree(self, t):
		p = weighted_sample(self.proposals, probs=self.probs, log=False)
		
		return p.propose_tree(t)

class InverseInliningProposal(LOTProposal):
	
	def __init__(self, grammar, replacetype='EXPR'):
		self.update(locals())
		LOTProposal.__init__(self, grammar)
		
	
	def propose_tree(self, t):
		
		
		## Insert/delete must happen at the apply nodes for FunctionGrammarInduction
		#G.add_rule('EXPR', 'apply_', ['LAMBDAARG', 'LAMBDATHUNK'], 1)
		#G.add_rule('LAMBDAARG',   'lambda', ['EXPR'], 1., bv_type='EXPR', bv_args=[] )
		#G.add_rule('LAMBDATHUNK', 'lambda', ['EXPR'], 1., bv_type=None, bv_args=None ) # A thunk

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
		
		if random() < 0.5:
			
			is_replacetype = lambda x: x.returntype is replacetype
			
			# sample a node
			for ni, di, resample_p, Z in self.grammar.sample_node_via_iterate(newt, do_bv=True, predicate=is_replacetype):
				
				# sample a subnode
				for s, sdi, sresample_p, sZ in self.grammar.sample_node_via_iterate(ni, do_bv=True, predicate=is_replacetype):
					
					# Now convert into a lambda abstraction
					
					pass
			
			# If we apply a lambda
			
			# Sample a lambda
			
			
			pass
			
		else:
			
			pass
		
		
		pass
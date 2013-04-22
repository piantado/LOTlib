# -*- coding: utf-8 -*-
import numpy as np

from copy import deepcopy
from random import randint

from LOTlib.Miscellaneous import *
from LOTlib.FunctionNode import *
#from LOTlib.FiniteBestSet import *

# A standard PCFG class. This should allow bound variables but as of 2011 Feb 2, these have not been thoroughly tested. 
# The args to a function must each either be terminals or FunctionNodes
# NOTE: We cleverly store each "rule" as a function node. As we add rules, we set their "rule id", with negative numbers for bound variables
#

"""


"""


class PCFG:
	
	def __init__(self):
		self.rules = dict()
		self.rule_count = 0
		self.bv_count = -1
		self.BV_WEIGHT = 10.0 # For now, this is a probability of the BV terminals TODO: Make nicer
	
	## HMM Nonterminals are only things that as in self.rules; ther ear ecomplex rules that are neither "nonterminals" by this, and terminals
	# nonterminals are those things that hash into rules
	def is_nonterminal(self, x): return (not islist(x)) and (x in self.rules)
	def is_terminal(self, x):    
		"""
			A terminal is not a nonterminal and either has no children or its children are terminals themselves
		"""
		if self.is_nonterminal(x): return False
		
		if isinstance(x, FunctionNode): 
			# else we must have
			for k in x.args:
				if not self.is_terminal(k): return False
		
		# else we get here for strings, etc.
		return True
		
	def display_rules(self):
		for k in self.rules.keys():
			print k
			for r in self.rules[k]:
				print "\t", r.as_list()
	
	def nonterminals(self):
		return self.rules.keys()
		
	# these take probs instead of log probs, for simplicity
	def add_rule(self, t, n, a, p, bv=[]):
		"""
			Adds a rule, and returns the added rule (for use by add_bv)
		"""
		if t not in self.rules: 
			self.rules[t] = [] # initialize to an empty list, so we can append
			
		# keep track of the rule counting
		if n.lower() == 'lambda':
			self.bv_count += 1
			
			if len(a) != 1: 
				print "***ERROR: lambda must have exactly one argument:", t, n, a, p, bv
				quit()
				
			if not isinstance(bv, list):
				print "*** ERROR: bound variables must be a list"
				quit()

		newrule = FunctionNode(t,n,a,log(p), bv=bv, rid=self.rule_count)
		
		self.rule_count += 1	
		
		self.rules[t].append(newrule)
		
		return newrule
	
	# takes a bit and expands it if its a nonterminal
	def sample_rule(self, f):
		if (f in self.rules):
			return weighted_sample(self.rules[f], return_probability=True) # get an expansion
		else: return [f,0.0]
		
	############################################################
	## Bound variable rules
	############################################################
	
	# removes the rule. This will do nothing if r is None
	def remove_rule(self, r):
		if r is not None:
			#print ">",self.rules[r.runturntype]
			self.rules[r.returntype].remove(r)
			#print ">>",self.rules[r.runturntype]
			
	# add a bound variable and return the rule
	def add_bv(self, t, d):
		"""
			Add an expansion to a bound variable of type t, at depth d
		"""
		if t is not None: 
			return self.add_rule(t, "y"+str(d), [], self.BV_WEIGHT)
		else: return None
			
	
	############################################################
	## generation
	############################################################
	
	# calls generate N times
	def generator(self, x, N):
		"""
			Call the generator N times -- an iterator
		"""
		for i in range(N):
			yield self.generate(x)
			
	# recursively sample rules
	# exanding the expansions of "to"
	# TODO: Set the max depth for generation
	def generate(self, x='START', d=0):
		"""
			Generate from the PCFG -- default is to start from 'START'
		"""
		
		if isinstance(x, FunctionNode): 
			
			addedrules = []
			for b in x.bv:
				addedrules.append( self.add_bv(b, d) ) # add bound variable
			
			f = x.copy()
			f.bv = addedrules
			f.args = [ (self.generate(k, d=d+1) if self.is_nonterminal(k) else k) for k in x.args ] # update the kids
			
			for r in addedrules:
				self.remove_rule(r) # remove bv rule
			
			return f
		elif self.is_nonterminal(x): # just a single thing   
			r,p = self.sample_rule(x)
			#print "\tRule: ", p, "\t", r
			n = self.generate(r, d=d+1)
			if isinstance(n, FunctionNode): n.lp = p # update the probability of this rule
			return n
		else:   return None
		
		
	# iterate through the subnodes of t, but updating my own bound variables to be 
	# accurate. Thus we can iterate up to some point and then have an accuraate PCFG
	# NOTE: if you DON'T iterate all the way through, you end up acculmulating bv rules
	# so NEVER stop this iteration in the middle!
	def iterate_subnodes(self, t, d=0):
		"""
			Iterate through all subnodes of t
		"""
		yield t
		
		for i in range(len(t.args)): # loop through kids
			if isinstance(t.args[i],FunctionNode):
				
				addedrules = []
				for b in t.bv:
					addedrules.append( self.add_bv(b, d) ) # add bound variable
				
				for ssn in self.iterate_subnodes(t.args[i], d+1):
					yield ssn
					
				for r in addedrules: self.remove_rule(r) # remove bv rule
			

	# choose a node at random and resample it
	def resample_random_subnode(self, t):
		"""
			resample a random subnode of t, returning a copy
		"""
		snc = randint(0, t.count_subnodes()-1) # pick the node to replace
		
		# copy since we modify in place
		newt = t.copy()
		
		i = -1 # what node are we on?
		for n in self.iterate_subnodes(newt):
			i = i + 1
			if i == snc:
				n.resample(self) # resample yourself
			# NOTE: Here you MUST evaluate on each loop iteration, or else this wont' remove the added bvrules -- no breaking!
			
		return newt
	
	# propose to t, returning [hyp, fb]
	def propose(self, t):
		"""
			propose to t by resampling a node. This returns [newtree, fb] where fb is forward log probability - backwards log probability
		"""
		newt = self.resample_random_subnode(t)
		fb = (-log(t.count_subnodes()) + newt.log_probability()) - ((-log(newt.count_subnodes())) + t.log_probability())
		return newt, fb
	
	
	## TODO: TEST THIS
	# this proposes but with a symmetric transition kernel
	# by doing rejection sampling on the proposals
	# mainly for debugging
	def propose_symmetric(self, t):
		"""
			Rejection sample on propose, so that fb is 0.0
		"""
		while True:
			newt, fb = self.propose(t)
			
			if random() > 1.0/(1.0+exp(-fb)): return [newt, 0.0]
	
	
	# yeild all pointwise changes to this function. this changes each function, trying all with the same type signature
	# and then yeilds the trees
	def enumerate_pointwise(self, t):
		"""
			Returns a generator of all the ways you can change a function (keeping the types the same) for t. Each gneeration is a copy
		"""
		for x in make_generator_unique(self.enumerate_pointwise_nonunique(t)):
			yield x
			
	# this enumerates using rules, but it may over-count, creating more than one instance. So we have to wrap it in 
	# a filter above
	def enumerate_pointwise_nonunique(self, t):
		for ti in t:
			titype = ti.get_type_signature() # for now, keep terminals as they are
			weightsum = logsumexp([ x.lp for x in self.rules[ti.returntype]])
			old_name, old_lp = [ti.name, ti.lp] # save these to restore
			possible_rules = filter(lambda ri: (ri.get_type_signature() == titype), self.rules[ti.returntype])
			if len(possible_rules) > 1:  # let's not yeild ourselves in all the ways we can
				for r in possible_rules: # for each rule of the same type
					# add this rule, copying over
					ti.name = r.name
					ti.lp = r.lp - weightsum # this is the lp -- the rule was unnormalized
					yield t.copy() # now the pointers are updated so we can yield this
			ti.name, lp = [old_name, old_lp]
		
		
	def increment_tree(self, x, depth):
		""" 
			A lazy version of tree enumeration. Here, we generate all trees, starting from a rule or a nonterminal symbol. 
			
			This is constant memory
		"""
		
		if self.bv_count > -1:
			print "*** Error: increment_tree not yet implemented for bound variables."
			quit()
		
		if isinstance(x, FunctionNode) and depth >= 0: 
			#print "FN:", x, depth
			
			addedrules = []
			for b in x.bv:
				addedrules.append( self.add_bv(b, d) ) # add bound variable
				
			#if addedrule is not None: print "ADDED=",addedrule, addedrule.returntype
			original_x = x.copy()
			
			# go all odometer on the kids below::
			
			iters = [self.increment_tree(y,depth-1) if self.is_nonterminal(y) else None for y in x.args]
			if len(iters) == 0: yield x.copy()
			else:
				
				# First, initialize the arguments
				for i in xrange(len(iters)):
					if iters[i] is not None: x.args[i] = iters[i].next()
				
				# the index of the last terminal symbol (may not be len(iters)-1),
				last_terminal_idx = max( [i if iters[i] is not None else -1 for i in xrange(len(iters))] )
				
				## Now loop through the args, counting them up
				continue_counting = True
				while continue_counting: # while we continue incrementing
					
					yield x.copy() # yield the initial tree, and then each successive tree
					
					# and then process each carry:
					for carry_pos in xrange(len(iters)): # index into which tree we are incrementing
						if iters[carry_pos] is not None: # we are not a terminal symbol (mixed in)
							
							#print "\ti=",carry_pos, x
							
							try: 
								x.args[carry_pos] = iters[carry_pos].next()
								break # if we increment successfully, no carry, so break out of i loop
							except StopIteration: # if so, then "carry"								
								if carry_pos == last_terminal_idx: 
									continue_counting = False # done counting here
								elif iters[carry_pos] is not None:
									# reset the incrementer since we just carried
									iters[carry_pos] = self.increment_tree(original_x.args[carry_pos],depth-1)
									x.args[carry_pos] = iters[carry_pos].next() # reset this
									# and just continue your loop over i (which processes the carry)
				
			#print "REMOVING", addedrule
			for r in addedrules: self.remove_rule(r) # remove bv rule
			
		elif self.is_nonterminal(x): # just a single nonterminal  
			#print "NT:", x, depth
			
			## TODO: somewhat inefficient since we do this each time:
			## Here we change the order of rules to be terminals *first*
			## else we don't enumerate small to large
			terminals = []
			nonterminals = []
			for k in self.rules[x]:
				if self.is_terminal(k): terminals.append(k)
				else:                   nonterminals.append(k)
			
			Z = logsumexp([ r.lp for r in self.rules[x]] ) # normalizer
			
			if depth >= 0:
				# yield each of the terminals
				
				for t in terminals:
					n = t.copy()
					n.lp = n.lp - Z
					yield n
			
			if depth > 0:
				# and expand each nonterminal, if any
				if len(nonterminals) > 0: # if we have any rules to expand (maybe not, due to depth)
					for g in nonterminals:
						for n in self.increment_tree(g.copy(), depth-1):
							if isinstance(n, FunctionNode): 
								n.lp = n.lp - Z
							yield n
		else:   raise StopIteration
			
	def get_rule_counts(self, t):
		"""
			A list of vectors of counts of how often each nonterminal is expanded each way
			
			TODO: This is probably not super fast since we use a hash over rule ids, but
			      it is simple!
		"""
		
		counts = dict() # a count for each hash type
				
		# make a list of not
		if not isinstance(t, list): t = [t]
		
		for ti in t:
			for x in ti:
				if x.ruleid >= 0: 
					counts[x.ruleid] = counts.get(x.ruleid,0)+1
		
		# and convert into a list of vectors (with the right zero counts)
		out = []
		for nt in self.rules.keys():
			v = np.array([ counts.get(r.ruleid,0) for r in self.rules[nt] ])
			out.append(v)
		return out
		
	def RR_prior(self, t, prior=1.0):
		"""
			Compute the rational rules prior from Goodman et al. 
			
			NOTE: This has not yet been extensively debugged, so use with caution
			
			TODO: Add variable priors (different vectors, etc)
		"""
		lp = 0.0
		
		for c in self.get_rule_counts(t):
			theprior = np.repeat(prior,len(c))
			lp += (beta(c+theprior) - beta(theprior))
		return lp
	
	def all_simple_uncompositions(self, t):
		"""
			Yeild all uncompositions of this, meaning all ways it could have been composed from
			"simple" lambda expressions (function application, etc)
			
			So this takes a tree like (+ (1 3) 1) and yeilds
			
			(lambda y (y (1 3) 1)) +
			(lambda x 1) (lambda y (+ (x 3) 1)) 1
			(lambda x 1) (lambda y (+ (x 3) x)) 1
			etc.
			
			NOTE: This therefore creates lambdas with bound variables. 
			
			It is not really clear how to handle lambdas here -- can we just generate them willy-nilly? Do they have to be allowed in the grammar?
			
			TODO: This function is not well tested. It is mainly experimental. 
		"""
		
		# first get a count of all subtrees
		leaf_counts = dict()
		function_counts = dict()
		for x in t:
			leaf_counts[x] = leaf_counts.get(x,0) + 1
			if x.is_function():
				function_counts[x.name] = function_counts.get(x.name,0) + 1
		
		##
		# First yield all tree leaves
		##
		
		# now for each subtree t, which occurs n times, 
		# compute all subset replacements (e.g. we can replace each occurance or not)
		for x, numoccurances in leaf_counts.items():
			# TODO: WE NEED TO NOT INCLUDE BOUND VARIABLES HERE
			for N in xrange(1,2**numoccurances): 
				newt = t.copy()
				for y in newt:
					if y == x:
						if N & 0x1: # replace this one 
							y.name = "Z"
							y.args = []
							y.lp = 0.0
							y.bv = []
							y.ruleid = 0
						N = (N >> 1)
				
				# TODO: THIS IS WHERE WE WOULD CHECK TO HANDLE INSERTING LAMDAS
				#newt =  FunctionNode( newt.returntype, 'lambda', [newt], 1.0, bv=[])
				yield newt, x
		
		for x, numoccurances in function_counts.items():
			for N in xrange(1,2**numoccurances): 
				newt = t.copy()
				for y in newt:
					if y.name == x:
						if N & 0x1: # replace this one 
							y.name = "Z"
						N = (N >> 1)
				
				# TODO: THIS IS WHERE WE WOULD CHECK TO HANDLE INSERTING LAMDAS
				#newt =  FunctionNode( newt.returntype, 'lambda', [newt], 1.0, bv=[])
				yield newt, x
		
		##
		# Now yield all function compositions
		##
			

		
		
		
		
		
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
if __name__ == "__main__":
	
	
	print "Testing bvPCFG: (TO IMPLEMENT)"
	
	
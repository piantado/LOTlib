# -*- coding: utf-8 -*-
from numpy import *

from copy import deepcopy
from random import randint

from LOTlib.Miscellaneous import *
from LOTlib.FunctionNode import *
from LOTlib.PriorityQueue import *

# A standard PCFG class with no bound variables
# the Node probabilities store log probabilities
# The args to a function must each either be terminals or FunctionNodes
# TODO: We can hash each rule type signature too so that we can easily access other types

class PCFG:
	def __init__(self):
		self.rules = dict()
	
	## HMM Nonterminals are only things that as in self.rules; ther ear ecomplex rules that are neither "nonterminals" by this, and terminals
	# nonterminals are those things that hash into rules
	def is_nonterminal(self, x): return (not islist(x)) and (x in self.rules)
	def is_terminal(self, x):    
		"""
			A terminal is not a nonterminal and either has no children or its children are terminals themselves
		"""
		if self.is_nonterminal(x): return False
		
		# else we must have
		for k in x.args:
			if not self.is_terminal(k): return False
		
		return True
	
	def nonterminals(self):
		return self.rules.keys()
		
	# these take probs instead of log probs, for simplicity
	def add_rule(self, t, n, a, p, bvtype=None):
		if not t in self.rules: self.rules[t] = [] # initialize to an empty list, so we can append
		self.rules[t].append(FunctionNode(t,n,a,log(p), bvtype=bvtype))
	
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
			self.rules[r.returntype].remove(r)
	
	# add a bound variable and return the rule
	def add_bv(self, t, d):
		if t is not None: 
			addedrule = self.get_bv_rule(t, d)
			self.rules[t].append(addedrule)
			return addedrule
		else: return None
		
	# make a rule for bound variables
	def get_bv_rule(self, t, d): 
		return FunctionNode(t, None, ["y"+str(d)], lp=0.0)
	
	
	############################################################
	## generation
	############################################################
	
	
	#def EXPERIMENTAL_priority_queue_sampler(self, heuristic, N=-1):
		#"""
			#Each nonterminal has a priority queue of length at most N storing the best ways to expand it 
			#according to heuristic. We sample from those when we need one, and add it to the priority queue
			
			#TODO: This was never finished. Do not use. 
		#"""
		
		#queues = dict()
		#rule_lp = dict() # hash each rule to a log probability
		#for nt in self.rules.keys():
			#queues[nt] = PriorityQueue(N=N, max=True)
			
			## base case: push on all of the rules
			#[ queues[nt].push(r, heuristic(r)) for r in self.rules[nt] ]
			
			#Z = logsumexp( [ v.lp for v in self.rules[nt] ] )
			#for v in self.rules[nt]:
				#rule_lp[v] = v.lp - Z
		
		## create a sampler that uses the heuristic
		#def mysampler(x, d=0): 
			#""" 
				#A sampler we will return -- very similar to generate code
			#"""
			##for nt in self.rules.keys():
				##print nt, "-->", [ (v.value, v.priority) for v in queues[nt].Q]
			##print "\n\n"
			

			#if isinstance(x, FunctionNode): 
				
				#addedrule = self.add_bv(x.bvtype, d) # add bound variable
				
				#f = deepcopy(x)
				#f.args = [ mysampler(k, d=d+1) for k in x.args ] # update the kids
				
				
				##print ">>", f,(f not in queues[f.returntype]),  heuristic(f), type(f), isinstance(f, FunctionNode),f.returntype, f.returntype == "WORD"
				
				#if (f not in queues[f.returntype]):
					##print "H12", f, f.returntype, heuristic(f)
					#queues[f.returntype].push(f, heuristic(f)) # update the queue to keep this guy
				
				#self.remove_rule(addedrule) # remove bv rule
				
				#return f
			#elif self.is_nonterminal(x): # just a single thing   
				
				#r = queues[x].sample(log=True)
				
				## HMM NOT QUITE RIGHT BUT OKAYISH:
				#if r in rule_lp: r.lp = rule_lp[r]
				
				#n = mysampler(r, d=d+1) # recurse
				##print ">>", n.lp, n
				## TODO: FIX THE PROB OF EACH NODE HERE
				##if isinstance(n, FunctionNode): n.lp = p # update the probability of this rule
				#return n
			#else:   return x
		
		#return mysampler	
	
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
	def generate(self, x, d=0):
		"""
			Generate from the PCFG
		"""
		
		if isinstance(x, FunctionNode): 
			
			addedrule = self.add_bv(x.bvtype, d) # add bound variable
			
			f = deepcopy(x)
			f.args = [ self.generate(k, d=d+1) for k in x.args ] # update the kids
			
			self.remove_rule(addedrule) # remove bv rule
			
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
				
				addedrule = self.add_bv(t.bvtype, d) # add bound variable
				
				for ssn in self.iterate_subnodes(t.args[i], d+1):
					yield ssn
					
				self.remove_rule(addedrule) # remove bv rule

	# choose a node at random and resample it
	def resample_random_subnode(self, t):
		"""
			resample a random subnode of t, returning a copy
		"""
		snc = randint(0, t.count_subnodes()-1) # pick the node to replace
		
		# copy since we modify in place
		newt = deepcopy(t)
		
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
	# mainly for debuggin
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
					yield deepcopy(t) # now the pointers are updated so we can yield this
			ti.name, lp = [old_name, old_lp]
	
	def all_trees(self, start, depth):
		"""
			Generate all trees up to the depth. This yields all smaller trees too!
		"""
		
		# hash each nonterminal type to a list of its terminals
		# this is successively expanded 
		type_to_terminals = dict() 
		for nt in self.rules.keys():
			for r in self.rules[nt]:
				if self.is_terminal(r):
					type_to_terminals[nt].append(r)
					if self.start == nt: yield r.copy() # first, yield terminals
		
		# then successively build up and yield
		for d in xrange(depth):
			
			
			
			for nt in self.rules.keys():
				for r in self.rules[nt]:
					
		
	#def next_tree(self, t):
		"""
			This is essentially an enumerator over *all* trees, 
		"""

# -*- coding: utf-8 -*-
"""
	A demo that include almost everything for running the number model
"""
from LOTlib.Objects import *
from LOTlib.Miscellaneous import *
from LOTlib.BasicPrimitives import *

from random import randint
from copy import deepcopy

DATA_SIZE = 250
STEPS = 100000

ALPHA = 0.75
GAMMA = -20.0 # the log probability penalty for recursion
LG_1MGAMMA = log(1.0-exp(GAMMA)) # NOTE: not numerically great
USE_RR_PRIOR = True # if false, we just use log probability

#################################################
# A function node for n-ary trees
#################################################

class FunctionNode:
	"""
	
	"""
	
	def __init__(self, t, n, a, lp=0.0, bvtype=None, rid=None):
		self.returntype = t
		self.name = n
		self.args = a
		self.lp = lp
		self.bvtype = bvtype
		self.ruleid = rid
		
	# make all my parts the same as q (not copies)
	def setto(self, q):
		self.returntype = q.returntype
		self.name = q.name
		self.args = q.args
		self.lp = q.lp
		self.bvtype = q.bvtype
		self.ruleid = q.ruleid
	
	def copy(self):
		"""
			A more efficient copy that mainly copies the nodes
		"""
		newargs = [x.copy() if isinstance(x, FunctionNode) else deepcopy(x) for x in self.args]
		return FunctionNode(self.returntype, self.name, newargs, self.lp, self.bvtype, self.ruleid)
	
	# output a string that can be evaluated by python
	## NOTE: Here we do a little fanciness -- with "if" -- we convert it to the "correct" python form with short circuiting instead of our fancy ifelse function
	def pystring(self): 
		if self.args == []: # a terminal
			return str(self.name)
		elif self.name == "if_": # this gets translated
			return '(' + str(self.args[1]) + ') if (' + str(self.args[0]) + ') else (' + str(self.args[2]) + ')'
		else: return self.name+'('+' '+commalist( [ str(x) for x in self.args])+' )'
	
	# NOTE: in the future we may want to change this to do fancy things
	def __str__(self): return self.pystring()

	def __repr__(self): return self.pystring()

	def __hash__(self): return hash(str(self))

	def __cmp__(self, x): return cmp(str(self), str(x))

	def __eq__(self, other): return (cmp(self, other) == 0)
	
	def log_probability(self):
		"""
			Returns the log probability of this node. This is computed by the log probability of each argument,
			UNLESS "my_log_probability" is defined, and then it returns that
		"""
		lp = self.lp # the probability of my rule
		for i in range(len(self.args)):
			if isinstance(self.args[i], FunctionNode):
				#print "\t<", self.args[i], self.args[i].log_probability(), ">\n"
				lp = lp + self.args[i].log_probability() # plus all children
		return lp

	# use generator to enumerate all subnodes
	# NOTE: To do anything fancy, we should use PCFG.iterate_subnodes in order to update the grammar, resample, etc. 
	def all_subnodes(self):

		yield self  # I am a subnode of myself
		
		for i in range(len(self.args)): # loop through kids
			if isinstance(self.args[i],FunctionNode):
				for ssn in self.args[i]:
					yield ssn

	# resample myself from some grammar
	def resample(self, g):
		self.setto(g.generate(self.returntype))
		
	def contains_function(self, x):
		"""
			Check if this contains x as function below
		"""
		for n in self:
			if n.name == x: return True
		return False
	
	def count_nodes(self): return self.count_subnodes()

	def count_subnodes(self):
		c = 0
		for n in self: 
			c = c + 1
		return c
	



#################################################
# A PCFG class
#################################################

class PCFG:
	def __init__(self):
		self.rules = dict()
		self.rule_count = 0
		self.bv_count = -1
	
	## HMM Nonterminals are only things that as in self.rules; ther ear ecomplex rules that are neither "nonterminals" by this, and terminals
	# nonterminals are those things that hash into rules
	def is_nonterminal(self, x): return (not islist(x)) and (x in self.rules)
		
	def is_terminal(self, x):    
		""" Check conditions for something to be a terminal """
		
		# Nonterminals are not terminals
		if self.is_nonterminal(x): return False
		
		if isFunctionNode(x): 
			# You can be a terminal if you are a function with all non-FunctionNode arguments
			return not any([ isFunctionNode(xi) for xi in None2Empty(x.args)])
		else:
			return True # non-functionNodes must be terminals
	
	def nonterminals(self):
		return self.rules.keys()
		
	# these take probs instead of log probs, for simplicity
	def add_rule(self, t, n, a, p, bvtype=None):
		"""
			Adds a rule, and returns the added rule (for use by add_bv)
		"""
		if not t in self.rules: 
			self.rules[t] = [] # initialize to an empty list, so we can append
			
		# keep track of the rule counting
		if bvtype is None:
			newrule = FunctionNode(t,n,a,log(p), bvtype=bvtype, rid=self.rule_count)
			self.rule_count += 1
		else: 
			newrule = FunctionNode(t,n,a,log(p), bvtype=bvtype, rid=self.bv_count)
			self.bv_count -= 1
		self.rules[t].append(newrule)
		
		return newrule
	
	# takes a bit and expands it if its a nonterminal
	def sample_rule(self, f):
		if (f in self.rules):
			return weighted_sample(self.rules[f], return_probability=True) # get an expansion
		else: return [f,0.0]
		
	# recursively sample rules
	# exanding the expansions of "to"
	def generate(self, x, d=0):
		"""
			Generate from the PCFG
		"""
		
		if isinstance(x, FunctionNode): 
			
			# addedrule = None # Temporarily disabled here
			
			f = x.copy()
			f.args = [ self.generate(k, d=d+1) for k in x.args ] # update the kids
			
			#self.remove_rule(addedrule) # remove bv rule # temporarily disabled
			
			return f
		elif self.is_nonterminal(x): # just a single thing   
			r,p = self.sample_rule(x)
			#print "\tRule: ", p, "\t", r
			n = self.generate(r, d=d+1)
			if isinstance(n, FunctionNode): n.lp = p # update the probability of this rule
			return n
		else:   return None

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
				
				#addedrule = self.add_bv(t.bvtype, d) # add bound variable # Temporarily disabled
				
				for ssn in self.iterate_subnodes(t.args[i], d+1):
					yield ssn
					
				#self.remove_rule(addedrule) # remove bv rule # temporarily disabled
				
	# propose to t, returning [hyp, fb]
	def propose(self, t):
		"""
			propose to t by resampling a node. This returns [newtree, fb] where fb is forward log probability - backwards log probability
		"""
		newt = self.resample_random_subnode(t)
		fb = (-log(t.count_subnodes()) + newt.log_probability()) - ((-log(newt.count_subnodes())) + t.log_probability())
		return newt, fb
		
#################################################
# Define a number hypothesis 
#################################################

class NumberExpression():
	
	def __init__(self,  v=None): 
		self.set_value(v) # to zero out prior, likelhood, lp
		self.prior, self.likelihood, self.lp = [-Infinity, -Infinity, -Infinity] # this should live here in case we overwrite self_value
		
		if v is None: self.set_value(grammar.generate('WORD'))
		else:         self.set_value(v)
	
	# use this because it updates prior, likelihood, and lp
	def set_value(self, v):
		self.value = v
		
	def copy(self):
		"""
			Must define this else we return "hypothesis" as a copy
		"""
		return NumberExpression(v=self.value.copy())
		
	def propose(self): 
		p = self.copy()
		ph, fb = grammar.propose(self.value)
		p.set_value(ph)
		return p, fb
		
	def compute_prior(self): 
		"""
		
		"""
		if self.value.count_subnodes() > 15: # don't allow more than this many nodes
			self.prior = -Infinity
		else: 
			if self.value.contains_function("L_"): recursion_penalty = GAMMA
			else:                                  recursion_penalty = LG_1MGAMMA
			
			self.prior = (recursion_penalty + self.value.log_probability()) 
			
			self.lp = self.prior + self.likelihood
			
		return self.prior
	
	def get_function_responses(self, data):
		"""
			Return the response of myself to *everything* in data, a list
		"""
		f = evaluate_expression(self.value, recurse="L_")
		out = []
		for di in data:
			w,s = di
			try: fs = f(s)
			except: fs = 'undef' # catch recursion mess up -- this just never equals a word. 
			out.append(fs)
		#print out
		return out
		
	def compute_likelihood(self, data):
		"""
			Computes the likelihood of data.
			We used to cache the function, but that gets messy with pickling and memory management, so now
			we don't
			
			TODO: Make sure this precisely matches the number paper. 
			
		"""
		lp = 0.0
		responses = self.get_function_responses(data) # get my response to everything
		for r, di in zip(responses, data):
			w = di[0]
			if r == 'undef' or r is None: 
				lp += log(1.0/10.0) # if undefined, just sample from a base distribution
			else:   lp += log( (1.0 - ALPHA)/10.0 + ALPHA * ( r == w ) )
			
		self.likelihood = lp
		self.lp = self.prior + self.likelihood
		
		return lp
	
	def compute_posterior(self, d):
		p = self.compute_prior()
		l = self.compute_likelihood(d)
		return [p,l]
		
	# given a set, choose an utterance
	def sample_utterance(self, s):
		
		
		f = evaluate_expression(self.value, recurse="L_") 
		
		if random() < ALPHA: 
			return f(s)
		else: 
			return weighted_sample( ['one_', 'two_', 'three_', 'four_', 'five_', 'six_', 'seven_', 'eight_', 'nine_', 'ten_'] )
	
	## These are just handy:
	
	def __hash__(self): return hash(str(self.value))

	def __eq__(self, other): return self.value.__eq__(other.value)

	def __str__(self): return str(self.value)

	def __repr__(self): return str(self)

	def __cmp__(self, other): return cmp(self.value,other)



#################################################
## Define the grammar
#################################################

grammar = PCFG()

grammar.add_rule('BOOL', 'and_',    ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'or_',     ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'not_',    ['BOOL'], 1.0)

grammar.add_rule('BOOL', 'True',    [], 1.0)
grammar.add_rule('BOOL', 'False',   [], 1.0)
grammar.add_rule('BOOL', 'equal_word_',   [], 1.0)

## note that this can take basically any types for return values
grammar.add_rule('WORD', 'if_',    ['BOOL', 'WORD', 'WORD'], 0.5)
grammar.add_rule('WORD', 'ifU_',    ['BOOL', 'WORD'], 0.5) # if returning undef if condition not met

grammar.add_rule('BOOL', 'cardinality1_',    ['SET'], 1.0)
grammar.add_rule('BOOL', 'cardinality2_',    ['SET'], 1.0)
grammar.add_rule('BOOL', 'cardinality3_',    ['SET'], 1.0)

grammar.add_rule('BOOL', 'equal_',    ['WORD', 'WORD'], 1.0)
grammar.add_rule('WORD', 'L_',        ['SET'], 1.0) 

grammar.add_rule('SET', 'x',     [], 10.0)

grammar.add_rule('SET', 'union_',     ['SET', 'SET'], 1.0)
grammar.add_rule('SET', 'intersection_',     ['SET', 'SET'], 1.0)
grammar.add_rule('SET', 'setdifference_',     ['SET', 'SET'], 1.0)
grammar.add_rule('SET', 'select_',     ['SET'], 1.0)

grammar.add_rule('WORD', 'next_', ['WORD'], 1.0)
grammar.add_rule('WORD', 'prev_', ['WORD'], 1.0)

grammar.add_rule('WORD', 'undef', [], 1.0)
grammar.add_rule('WORD', 'one_', [], 0.10)
grammar.add_rule('WORD', 'two_', [], 0.10)
grammar.add_rule('WORD', 'three_', [], 0.10)
grammar.add_rule('WORD', 'four_', [], 0.10)
grammar.add_rule('WORD', 'five_', [], 0.10)
grammar.add_rule('WORD', 'six_', [], 0.10)
grammar.add_rule('WORD', 'seven_', [], 0.10)
grammar.add_rule('WORD', 'eight_', [], 0.10)
grammar.add_rule('WORD', 'nine_', [], 0.10)
grammar.add_rule('WORD', 'ten_', [], 0.10)

def get_knower_pattern(ne):
	"""
		This computes a string describing the behavior of this knower-level
	"""
	# out = ''
	mydata = [ ('', set(sample_sets_of_objects(n, all_objects))) for n in xrange(1,10) ] 
	resp = ne.get_function_responses( mydata )
	return ''.join([ str(word_to_number[x]) if (x is not None and x is not 'undef' ) else 'U' for x in resp])


#################################################
## The target lexicon and sets of objects
#################################################

# How the parent generates words
target = NumberExpression("one_ if cardinality1_(x) else next_(L_(setdifference_(x, select_(x))))") # we need to translate "if" ourselves

#here this is really just a dummy -- one type of object, which is replicated in sample_sets_of_objects
all_objects = make_all_objects(shape=['duck'])  

# all possible data sets
all_possible_data = [ ('', set(sample_sets_of_objects(n, all_objects))) for n in xrange(1,10) ] # must NOT be just the pointers sampled, since then set() operations will collapse them!
	
	
# # # # # # # # # # # # # # # # # # # # # # # # #
# Run MCMC
# # # # # # # # # # # # # # # # # # # # # # # # #

# make up some random data
data = []
for i in range(DATA_SIZE):
	# how many in this set
	set_size = weighted_sample( range(1,10+1), probs=[7187, 1484, 593, 334, 297, 165, 151, 86, 105, 112] )
	# get the objects in the current set
	s = set(sample_sets_of_objects(set_size, all_objects))
	# and append the sampled utterance
	data.append( [ target.sample_utterance(s), s] )

hyp = NumberExpression()

current_sample = NumberExpression() # generate a number expression
for mhi in xrange(1000000): # how many steps
		
	p, fb = current_sample.propose() # a proposal and a forward-back probability
	np, nl = p.compute_posterior(data)
		
	r = (np+nl)-(current_sample.prior + current_sample.likelihood)-fb
	
	if r > 0.0 or random() < exp(r):  # keep the sample if its good
		current_sample = p
	
	# now we have a new sample--print it out
	print q(get_knower_pattern(current_sample)), current_sample.prior, current_sample.likelihood, q(current_sample)

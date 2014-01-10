# -*- coding: utf-8 -*-
"""
	A function node -- a tree part representing a function and its arguments. 
	Also used for PCFG rules, where the arguments are nonterminal symbols. 
	
	A Functionnode defaultly iterates over its subnodes
	
	TODO: This could use some renaming FunctionNode.bv is not really a bound variable--its a list of rules that were added
"""

import re

from copy import copy, deepcopy
from LOTlib.Miscellaneous import *

def list2FunctionNode(l, style="atis"):
	"""
		Take a list and map it to a function node. 
		This will take lambda arguments of some given style (e.g. atis, scheme, etc)
	"""
	
	if isinstance(l, list): 
		if len(l) == 0: return None
		elif style is 'atis':
			rec = lambda x: list2FunctionNode(x, style=style) # a wrapper to my recursive self
			if l[0] == 'lambda':
				return FunctionNode('FUNCTION', 'lambda', [rec(l[3])], lp=0.0, bv_type=[l[1]] ) ## TOOD: HMM WHAT IS THE BV?
			else:
				return FunctionNode(l[0], l[0], map(rec, l[1:]), lp=0.0, bv_type='')
		elif sytle is 'scheme':
			pass #TODO: Add this scheme functionality -- basically differnet handling of lambda bound variables
			
	else: # for non-list
		return l

# a regex for matching variables (y0,y1,.. y15, etc)
re_variable = re.compile(r"y([0-9]+)?$")

# just because this is nicer, and allows us to map, etc. 
def isFunctionNode(x): return isinstance(x, FunctionNode)

def cleanFunctionNodeString(x):
	"""
		Make functionNode strings easier to read
	"""
	s = re.sub("lambda", u"\u03BB", str(x)) # make lambdas the single char
	s = re.sub("_", '', s) # remove underscores
	return s
	

class FunctionNode(object):
	"""
		NOTE: If a node has [ None ] as args, it is treated as a thunk
		
		bv - stores the actual *rule* that was added (so that we can re-add it when we loop through the tree)
		
		My bv stores the particlar *names* of variables I've introduced
	"""
	
	def __init__(self, returntype, name, args, lp=0.0, resample_p=1.0, bv_name=None, bv_type=None, bv_args=None, ruleid=None):
		self.__dict__.update(locals())
		
	# make all my parts the same as q (not copies)
	def setto(self, q):
		self.__dict__.update(q.__dict__)
			
	def __copy__(self, shallow=False):
		"""
			Copy a function node
			shallow - if True, this does not copy the children (self.to points to the same as what we return)
		"""
		if (not shallow) and self.args is not None:
			newargs = [copy(x) if isFunctionNode(x) else deepcopy(x) for x in self.args]
		else:
			newargs = self.args
		
		return FunctionNode(self.returntype, self.name, newargs, lp=self.lp, resample_p=self.resample_p, bv_type=self.bv_type, bv_name=self.bv_name, bv_args=deepcopy(self.bv_args), ruleid=self.ruleid)
	
	def is_nonfunction(self):
		return (self.args is None)
	def is_function(self):
		return not self.is_nonfunction()
	
	def as_list(self):
		"""
			This returns ourself structured as a lisp (with function/self.name first)
			NOTE: This does ot handle BV yet
		"""
		x = [self.name] if self.name != '' else []
		if self.args is not None:
			x.extend( [a.as_list() if isFunctionNode(a) else a for a in self.args] )
		return x
			
	# output a string that can be evaluated by python
	## NOTE: Here we do a little fanciness -- with "if" -- we convert it to the "correct" python form with short circuiting instead of our fancy ifelse function
	def pystring(self): 
		#print ">>", self.name
		if self.is_nonfunction(): # a terminal
			return str(self.name)
		elif self.name == "if_": # this gets translated
			assert len(self.args) == 3, "if_ requires 3 arguments!"
			return '(' + str(self.args[1]) + ' if ' + str(self.args[0]) + ' else ' + str(self.args[2]) + ')'
			#return '(' + str(self.args[1]) + ') if (' + str(self.args[0]) + ') else (' + str(self.args[2]) + ')'
		elif self.name == '':
			assert len(self.args) == 1, "Null names must have exactly 1 argument"
			return str(self.args[0])
		elif self.name is not None and self.name.lower() == 'apply_':
			assert self.args is not None and len(self.args)==2, "Apply requires exactly 2 arguments"
			return '('+str(self.args[0])+')('+str(self.args[1])+')'
		elif self.name is not None and self.name.lower() == 'lambda':
			assert len(self.args) == 1, "Lambda variables require one argument"
			
			# We can allow bv_type to be None, which is a thunk (no arguments)
			#return '(lambda '+ (self.bv_type if self.bv_type is not None else '') +': '+str(self.args[0])+' )'
			
			# The old version (above) wrapped in parens, but that's probably not necessary?
			return 'lambda '+ (self.bv_name if self.bv_name is not None else '') +': '+str(self.args[0])
		else:
			
			if self.args is None:
				return str(self.name)+'()' # simple call
			else:
				return str(self.name)+'('+', '.join(map(str,self.args))+')'
	
	def quickstring(self):
		"""
			A (maybe??) faster string function used for hashing -- doesn't handle any details and is meant
			to just be quick
		"""
		if self.args is None:
			return str(self.name) # simple call
		else:
			return str(self.name)+' '+','.join(map(str,self.args)) 
		
	def fullprint(self, d=0):
		""" A handy printer for debugging"""
		tabstr = "  .  " * d
		print tabstr, self.returntype, self.name, self.bv_type, self.bv_name, self.bv_args, "\t", self.lp #"\t", self.resample_p 
		if self.args is not None:
			for a in self.args: 
				if isFunctionNode(a): a.fullprint(d+1)
				else:                 print tabstr, a
			
	#def schemestring(self):
		#if self.args == []: # a terminal
			#return str(self.name)
		#else: return '('+self.name + ' '+commalist( [ str(x) for x in self.args], sep1=' ', sep2=' ')+' )'
	
	# NOTE: in the future we may want to change this to do fancy things
	def __str__(self): return self.pystring()
	def __repr__(self): return self.pystring()
	
	def __eq__(self, other): return isFunctionNode(other) and (cmp(self, other) == 0)
	def __ne__(self, other): return not self.__eq__(other)
	
	## TODO: overwrite these with something faster
	# hash trees. This just converts to string -- maybe too slow?
	def __hash__(self):
		
		# An attempt to speed things up -- not so great!
		#hsh = self.ruleid
		#if self.args is not None:
			#for a in filter(isFunctionNode, self.args):
				#hsh = hsh ^ hash(a)
		#return hsh
		
		# normal string hash -- faster?
		return hash(str(self))
		
		# use a quicker string hash		
		#return hash(self.quickstring())
		
		
	def __cmp__(self, x): return cmp(str(self), str(x))
	
	def __len__(self): return len([a for a in self])
	
	def log_probability(self):
		"""
			Returns the log probability of this node. This is computed by the log probability of each argument,
			UNLESS "my_log_probability" is defined, and then it returns that
		"""
		if hasattr(self, 'my_log_probability'):
			#print ">!!>>", t, self.my_log_probability
			return self.my_log_probability
		else:
			lp = self.lp # the probability of my rule
			if self.args is None: return lp
			for i in range(len(self.args)):
				if isFunctionNode(self.args[i]):
					#print "\t<", self.args[i], self.args[i].log_probability(), ">\n"
					lp = lp + self.args[i].log_probability() # plus all children
			return lp
	
	# use generator to enumerate all subnodes
	# NOTE: To do anything fancy, we should use PCFG.iterate_subnodes in order to update the grammar, resample, etc. 
	def all_subnodes(self):
		print "*** USE __ITER__ now!"
		assert(False)
		
	def __iter__(self):
		
		yield self
		
		if self.args is not None:
			for a in filter(isFunctionNode, self.args):
				for ssn in a: yield ssn
	
	def all_leaves(self):
		if self.args is not None:
			for i in range(len(self.args)): # loop through kids
				if isFunctionNode(self.args[i]):
					for ssn in self.args[i].all_leaves():
						yield ssn
				else:
					yield self.args[i]

	def string_below(self, sep=" "):
		return sep.join(map(str, self.all_leaves()))
	
	def fix_bound_variables(self, d=0, rename=None):
		"""
			Fix the naming scheme of bound variables. This happens if we promote or demote some nodes
			via insert/delete
			
			d - current depth
			rename - a dictionary to store how we should rename
		"""
		if rename is None: rename = dict()
				
		if self.name is not None:
			if self.name.lower() == 'lambda' and (self.bv_type is not None) and (self.args is not None): 
				#assert (self.bv_args is None) # should only add one rule, and it should have no "to"
				
				newname = 'y'+str(d)
					
				# And rename this below
				rename[self.bv_type] = newname
				self.bv_type = newname
				#print "..", self.bv[0]
			elif re_variable.match(self.name): # if we find a variable
				assert_or_die(self.name in rename, "Name "+self.name+" not in rename="+str(rename)+"\t;\t"+str(self))
				self.name = rename[self.name]
		
		# and recurse
		if self.args is not None:
			for k in self.args:
				if isFunctionNode(k): k.fix_bound_variables(d+1, rename)
			

	############################################################
	## Derived functions that build on the above core
	############################################################
	
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
	
	def depth(self):
		depths = [0] # if no function nodes
		if self.args is not None:
			for i in range(len(self.args)):
				if isFunctionNode(self.args[i]):
					depths.append( self.args[i].depth()+1 )
		return max(depths)
	
	# get a description of the input and output types
	# if collapse_terminal then we just map non-FunctionNodes to "TERMINAL"
	def get_type_signature(self):
		ts = [self.returntype, self.bv_type, self.bv_args]
		if self.args is not None:
			for i in range(len(self.args)):
				if isFunctionNode(self.args[i]):
					ts.append(self.args[i].returntype)
				else: 
					ts.append(self.args[i])
		return ts
	
	def is_replicating(self):
		"""
			A function node is replicating (by definition) if one of its children is of the same type
		"""
		return any([isFunctionNode(x) and x.returntype == self.returntype for x in self.args if self.args is not None])
		

	def is_canonical_order(self, symmetric_ops):
		"""
			Takes a set of symmetric ops (plus, minus, times, etc, not divide) and asserts that the LHS ordering is less than the right (to prevent)
		"""
		if self.args is None or len(self.args) == 0: return True
		
		if self.name in symmetric_ops:
			
			# Then we must check children
			if self.args is not None:
				for i in xrange(len(self.args)-1):
					if self.args[i].name > self.args[i+1].name: return False
			
		# Now check the children, whether or not we are symmetrical
		return all([x.is_canonical_order(symmetric_ops) for x in self.args if self.args is not None])
		
	def proposal_probability_to(self, y):
		"""
			Proposal probability from self to y
		
			TODO: NOT HEAVILY TESTED/DEBUGGED. PLEASE CHECK
		"""
		
		# We could do this node:
		pself = -log(len(self))	
		
		if( self.returntype != y.returntype):
			
			return float("-inf")
		
		elif(self.name != y.name):
			
			# if names are not equal (but return types are) we must generate from the return type, using the root node
			return pself + y.log_probability() 
		
		else:
			# we have the same name and returntype, so we may generate children
			
			# Compute the arguments and see how mismatched we are
			mismatches = []
			if self.args is not None:
				for a,b in zip(self.args, y.args):
					if a != b: mismatches.append( [a,b] )
			
			# Now act depending on the mismatches
			if len(mismatches) == 0: # we are identical below
				
				# We are the same below here, so we can propose to any subnode, which 
				# each happens with prob pself
				return logsumexp( [pself + t.log_probability() for t in self] )
			
			elif len(mismatches) == 1:
				
				a,b = mismatches[0]
				
				# Well if there's one mismatch, it lies below a or b,
				# so we must propose in a along this subtree
				m = log(len(a)) + pself # choose uniformly from this subtree, as individual nodes are adjusted later TODO: IM NOT SURE THIS IS RIGHT 
				return logsumexp([m + a.proposal_probability_to(b) , pself + y.log_probability()])
			
			else: return pself + y.log_probability() # We have to generate from ourselves
				











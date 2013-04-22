# -*- coding: utf-8 -*-
"""
	A function node -- a tre part representing a function and its arguments. 
	Also used for PCFG rules, where the arguments are nonterminal symbols. 
"""

import re

from numpy import *
from scipy.maxentropy import logsumexp
from copy import deepcopy
from LOTlib.SimpleLambdaParser import *

from LOTlib.Miscellaneous import *



def list2FunctionNode(l, style="atis"):
	"""
		Take a list and map it to a function node. 
		This will take lambda arguments of some given style (e.g. atis, scheme, etc)
	"""
	#print l
	
	if isinstance(l, list): 
		if len(l) == 0: return None
		elif style is 'atis':
			rec = lambda x: list2FunctionNode(x, style=style) # a wrapper to my recursive self
			if l[0] == 'lambda':
				return FunctionNode('FUNCTION', 'lambda', [rec(l[3])], lp=0.0, bv=[l[1]] ) ## TOOD: HMM WHAT IS THE BV?
			else:
				return FunctionNode(l[0], l[0], map(rec, l[1:]), lp=0.0, bv=[])
		elif sytle is 'scheme':
			pass #TODO: Add this scheme functionality -- basically differnet handling of lambda bound variables
			
	else: # for non-list
		return l


class FunctionNode:
	"""
		NOTE: If a node has [ None ] as args, it is treated as a thunk
	"""
	
	def __init__(self, t, n, a, lp=0.0, bv=[], rid=None):
		self.returntype = t
		self.name = n
		self.args = a
		self.lp = lp
		self.bv = bv
		self.ruleid = rid
		
	# make all my parts the same as q (not copies)
	def setto(self, q):
		self.returntype = q.returntype
		self.name = q.name
		self.args = q.args
		self.lp = q.lp
		self.bv = q.bv
		self.ruleid = q.ruleid
	
	def copy(self):
		"""
			A more efficient copy that mainly copies the nodes
		"""
		newargs = [x.copy() if isinstance(x, FunctionNode) else deepcopy(x) for x in self.args]
		return FunctionNode(self.returntype, self.name, newargs, self.lp, self.bv, self.ruleid)
	
	def is_nonfunction(self):
		return (self.args is None) or (len(self.args)==0) or ( len(self.args)==1 and self.args[0] is None)
	def is_function(self):
		return not self.is_nonfunction()
	
	def as_list(self):
		"""
			Not a pretty print, just a list of all key feature
		"""
		return [self.returntype, self.name, self.args, self.lp, self.bv, self.ruleid]
	
	# output a string that can be evaluated by python
	## NOTE: Here we do a little fanciness -- with "if" -- we convert it to the "correct" python form with short circuiting instead of our fancy ifelse function
	def pystring(self): 
		#print self.name, "BV=", self.bv
		if self.is_nonfunction(): # a terminal
			return str(self.name)
		elif self.name == "if_": # this gets translated
			return '(' + str(self.args[1]) + ') if (' + str(self.args[0]) + ') else (' + str(self.args[2]) + ')'
		elif self.name.lower() == 'lambda':
			#print len(self.bv)
			return '(lambda '+commalist( [ str(x) for x in self.bv])+': '+str(self.args[0])+' )'
		else: 
			if len(self.args) == 1 and self.args[0] is None: # handle weird case with None as single terminal below
				return self.name+'()'
			else:
				return self.name+'('+' '+commalist( [ str(x) for x in self.args])+' )'
	
	#def schemestring(self):
		#if self.args == []: # a terminal
			#return str(self.name)
		#else: return '('+self.name + ' '+commalist( [ str(x) for x in self.args], sep1=' ', sep2=' ')+' )'
	
	
	
	# NOTE: in the future we may want to change this to do fancy things
	def __str__(self): return self.pystring()
	def __repr__(self): return self.pystring()
	
	def __eq__(self, other): return (cmp(self, other) == 0)
	def __ne__(self, other): return not self.__eq__(other)
	
	## TODO: overwrite these with something faster
	# hash trees. This just converts to string -- maybe too slow?
	def __hash__(self): return hash(str(self))
	def __cmp__(self, x): return cmp(str(self), str(x))
	
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
			for i in range(len(self.args)):
				if isinstance(self.args[i], FunctionNode):
					#print "\t<", self.args[i], self.args[i].log_probability(), ">\n"
					lp = lp + self.args[i].log_probability() # plus all children
			return lp
	
	# use generator to enumerate all subnodes
	# NOTE: To do anything fancy, we should use PCFG.iterate_subnodes in order to update the grammar, resample, etc. 
	def all_subnodes(self):

		yield self;  # I am a subnode of myself
		
		for i in range(len(self.args)): # loop through kids
			if isinstance(self.args[i],FunctionNode):
				for ssn in self.args[i]:
					yield ssn

	# resample myself from some grammar
	def resample(self, g):
		if g.is_nonterminal(self.returntype):
			self.setto(g.generate(self.returntype))
		else: pass # do nothing if we aren't returnable from the grammar
		
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
		depths = [0.0] # if no function nodes
		for i in range(len(self.args)):
			if isinstance(self.args[i],FunctionNode):
				depths.append( self.args[i].depth()+1 )
		return max(depths)
	
	# get a description of the input and output types
	# if collapse_terminal then we just map non-FunctionNodes to "TERMINAL"
	def get_type_signature(self):
		ts = [self.returntype, self.bv]
		for i in range(len(self.args)):
			if isinstance(self.args[i],FunctionNode):
				ts.append(self.args[i].returntype)
			else: 
				ts.append(self.args[i])
		return ts
	
		
		
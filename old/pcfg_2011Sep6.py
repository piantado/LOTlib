from numpy import *
from scipy.maxentropy import logsumexp

def lenr(x): 
	return range(len(x))

def islist(x): 
	return isinstance(x,list)


# takes unnormalized probabilities
def weighted_sample(objs, probs=None, log=False):
	
	# check how probabilities are specified
	# either as an argument, or attribute of objs (either probability or lp
	myprobs = None
	if probs is None:
		if hasattr(objs[0], 'probability'): # may be log or not
			myprobs = map(lambda x: x.probability, objs)
		elif hasattr(objs[0], 'lp'): # MUST be logs
			myprobs = map(lambda x: x.lp, objs)
			log = True 
	else: 
		myprobs = probs

	# Now normalize and run
	Z = None
	if log: Z = logsumexp(myprobs)
	else: Z = sum(myprobs)
		
	r = random.rand()
	for i in lenr(objs):
		if log: 
			r = r - exp(myprobs[i]-Z) # log domain
		else:     
			r = r - (myprobs[i]/Z) # probability domain
			
		if r <= 0: 
			return objs[i]
		
	
	return None # Maybe return the last since we shouldn't get here?
	
	
	
# this stores trees and rules. p is probability in non-log domain
# lp is probability in log domain
class Node:
	def __init__(self, f, to, lp=None):
		self.frm = f
		self.to = to
		self.lp = lp
	def __str__(self):
		tostr = ""
		if islist(self.to): tostr=str(map(str, self.to))
		else: tostr = str(self.to)
			
		return '('+self.frm +' '+tostr+' ~'+str(self.lp)+' )'


# A standard PCFG class with no bound variables
# the Node probabilities store log probabilities

class PCFG:
	
	def __init__(self):
		self.rules = dict()
		pass
	
	# nonterminals are those things that hash into rules
	def is_nonterminal(self, x): return (not islist(x)) and (x in self.rules)
	def is_terminal(self, x):    return not is_nonterminal(x)
	
	def add_rule(self, f, t, p):
		if not f in self.rules: self.rules[f] = [] # initialize to an empty list, so we can append
		self.rules[f].append(Node(f,t,log(p)))
	
	# takes a bit and expands it if its a nonterminal
	def sample_rule(self, f):
		if (not islist(f)) and (f in self.rules):
			return weighted_sample(self.rules[f]) # get an expansion
		else: return f	
	
	# recursively sample rules
	# exanding the expansions of "to"
	def recursive_generate(self, x):
		#print "Recursive generate: "+str(x)
		children = None
		children_lp = 0
		if isinstance(x, Node): 
			return Node(x.frm, self.recursive_generate(x.to), x.lp) # to generate from a Node/rule, expand the children
		elif islist(x):  # if you are a list, then map over the list and return a list
			return [ self.recursive_generate(self.sample_rule(xi)) for xi in x] 
		elif self.is_nonterminal(x): # just a single thing   
			return self.recursive_generate(self.sample_rule(x))
		else:   return x
		
	# take a tree and compile it to python
	def topython(self):
		
		
x = PCFG()
x.add_rule('EXPR', ['+', 'EXPR', 'EXPR'], 1.0)
x.add_rule('EXPR', ['/', 'EXPR', 'EXPR'], 1.0)
x.add_rule('EXPR', '1', 1.0)
x.add_rule('EXPR', '2', 1.0)
x.add_rule('EXPR', '3', 1.0)
x.add_rule('EXPR', '4', 1.0)
x.add_rule('EXPR', '5', 1.0)
x.add_rule('EXPR', '6', 1.0)

print x.recursive_generate('EXPR')			





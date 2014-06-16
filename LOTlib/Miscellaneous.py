# -*- coding: utf-8 -*-

"""
	Miscellaneous functions for LOTlib
	
	Steve Piantadosi - Sept 2011
"""


# Special handilng for numpypy that doesn't use gammaln, assertion error otherwise
try: 
	from scipy.special import gammaln
except ImportError: 
	# Die if we try to use this in numpypy
	def gammaln(*args, **kwargs): assert False


try:                import numpy as np
except ImportError: import numpypy as np


from random import random, sample, randint
import itertools
from math import exp, log, sqrt, pi, e
import functools # for memoize
import os
import sys
import math
import collections

import types # for checking if something is a function: isinstance(f, types.FunctionType)

## Some useful constants
Infinity = float("inf")
inf = Infinity
Inf = Infinity
Null = []
TAU = 6.28318530718 #fuck pi

## For R-friendly
T=True
F=False

def first(x): return x[0]
def second(x): return x[1]
def third(x):  return x[2]
def fourth(x):  return x[3]
def fifth(x):  return x[4]
def sixth(x):  return x[5]
def seventh(x):  return x[6]
def eighth(x):  return x[7]

def dropfirst(g):
	"""
		Return all but the first element
	"""
	keep = False
	for x in g:
		if keep: yield x
		keep = True

def None2Empty(x):
	# Treat Nones as empty
	if x is None: return []
	else:         return x

def make_mutable(x):
	# TODO: update with other types
	if isinstance(x, frozenset): return set(x)
	elif isinstance(x, tuple): return list(x)
	else: return x

def make_immutable(x):
	# TODO: update with other types
	if isinstance(x, set ): return frozenset(x)
	elif isinstance(x, list): return tuple(x)
	else: return x

def unlist_singleton(x):
	"""
		Remove any sequences of nested lists with one element. 
		
		e.g. [[[1,3,4]]] -> [1,3,4]
	"""
	if isinstance(x,list) and len(x) == 1:
		return unlist_singleton(x[0])
	else:
		return x

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


## for printing wiht debug levels. 
## Here we have a global variable (defaults to 10) we print all statements with l <= DEBUG_LEVEL
## Library internal debugging should have level >= 100, I think
from multiprocessing import Process # this forces prints to be flushed, so we can flush with dprint below to prevent threads from interrupting each other
global DEBUG_LEVEL
DEBUG_LEVEL = 10
def dprintn(l, *args):
	args = list(args)
	args.append("\n")
	dprint(l,*args)
def dprint(l, *args):
	global DEBUG_LEVEL
	if DEBUG_LEVEL >= l:
		for a in args: 
			print str(a),
	sys.stdout.flush()

def dprinterr(l, *args):
	global DEBUG_LEVEL
	if DEBUG_LEVEL >= l:
		for a in args: 
			print >>sys.stderr, str(a),
	sys.stderr.flush()

	
def fprintn(dl, *args, **kwargs):
	
	f = kwargs.get('f',sys.stdout)
	
	if DEBUG_LEVEL >= dl:
		o = open(f, 'a')
		for a in args: 
			print >>o, str(a),
		print >>o, "\n",
		if f is not sys.stdout: o.close()
		
def q(x, quote='\''): 
	if isinstance(x,str) or isinstance(x, unicode):
		return quote+x+quote
	else:
		return quote+str(x)+quote

def qq(x): return q(x,quote="\"")

def display(x): print x
	
# for functional programming, print something and return it
def printr(x):
	print x
	return x
	
def r2(x): return round(x,2)
def r3(x): return round(x,3)
def r4(x): return round(x,4)
def r5(x): return round(x,5)

def tf201(x):
	if x: return 1
	else: return 0


from time import gmtime, strftime, time, localtime
## Functions for I/O
def display_option_summary(obj):
	"""
		Prints out a friendly format of all options -- for headers of output files
		This takes in an OptionParser object as an argument. As in, (options, args) = parser.parse_args()
	"""
	
	print "####################################################################################################"
	
	try: print "# Username: ", os.getlogin()
	except OSError: pass
	
	try: print "# Date: ", strftime("%Y %b %d (%a) %H:%M:%S", localtime(time()) )
	except OSError: pass
	
	try: print "# Uname: ", os.uname()
	except OSError: pass
	
	try: print "# Pid: ", os.getpid()
	except OSError: pass
	
	for slot in dir(obj):
		attr = getattr(obj, slot)
		if not isinstance(attr, (types.BuiltinFunctionType, types.FunctionType, types.MethodType)) and (slot is not "__doc__") and (slot is not "__module__"):
			print "#", slot, "=", attr
	print "####################################################################################################"

	

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Genuine Miscellany
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def list2str(x, sep=' '):
	return sep.join(x)

# a wrapper so we can call this in the below weirdo composition
def raise_exception(e): raise e

def ifelse(x,y,z):
	if x: return y
	else: return z
	
# add to a hash list
def hashplus(d, k, v=1):
	if not k in d: d[k] = v
	else: d[k] = d[k] + v


def unique(gen):
	"""
		Make a generator unique, returning each element only once
	"""
	s = set()
	for gi in gen:
		if gi not in s:
			yield gi
			s.add(gi)
			
def UniquifyFunction(gen):
	"""
		A decorator to make a function only return unique values
	"""
	def f(*args, **kwargs):
		for x in unique(gen(*args, **kwargs)): 
			yield x
	return f
	
	
def listifnot(x):
	if isinstance(x,list): return x
	else:                  return [x]


def all_binary_vectors(N):
	return [  [ (x>>n)&0x1 for n in xrange(N)] for x in xrange(0,2**N) ]
	
	
def flatten(expr): 
	"""
		Flatten lists of lists, via stackoverflow
	"""
	def flatten_(expr): 
		#print 'expr =', expr
		if expr is None or not isinstance(expr, collections.Iterable) or isinstance(expr, str):
			yield expr
		else:
			for node in expr:
				#print node, type(node)
				if (node is not None) and isinstance(node, collections.Iterable) and (not isinstance(node, str)):
					#print 'recursing on', node
					for sub_expr in flatten_(node):
						yield sub_expr
				else:
					#print 'yielding', node
					yield node

	return tuple([x for x in flatten_(expr)])	

def flatten2str(expr, sep=' '):
	try:
		if expr is None: return ''
		else:            return sep.join(flatten(expr))
	except TypeError:
		print "Error in flatter2str:", expr
		raise TypeError

def weave(*iterables):
	"""
	Intersperse several iterables, until all are exhausted.
	This nicely will weave together multiple chains
	
	from: http://www.ibm.com/developerworks/linux/library/l-cpyiter/index.html
	"""
	
	iterables = map(iter, iterables)
	while iterables:
		for i, it in enumerate(iterables):
			try: yield it.next()
			except StopIteration: del iterables[i]	
		
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Math functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


## This is just a wrapper to avoid logsumexp([-inf, -inf, -inf...]) warnings
try:			
	from scipy.misc import logsumexp as scipy_logsumexp
except ImportError:	
	try:
		from scipy.maxentropy import logsumexp as scipy_logsumexp
	except ImportError:
		# fine, our own version, no numpy
		def scipy_logsumexp(v):
			m = max(v)
			return m+log(sum(map( lambda x: exp(x-m), v)))
			
def logsumexp(v):
	"""
		Logsumexp - our own version wraps the scipy to handle -infs
	"""
	if max(v) > -Infinity: return scipy_logsumexp(v)
	else: return -Infinity

def lognormalize(v):
	return v - logsumexp(v)

def logplusexp(a, b):
	"""
		Two argument version. No cast to numpy, so faster
	"""
	m = max(a,b)
	return m+log(exp(a-m)+exp(b-m))

def beta(a):
	""" Here a is a vector (of ints or floats) and this computes the Beta normalizing function,"""
	return np.sum(gammaln(np.array(a, dtype=float))) - gammaln(float(sum(a)))
		


def islist(x): return isinstance(x,list)


def normlogpdf(x, mu, sigma):
	""" The log pdf of a normal distribution """
	#print x, mu
	return math.log(math.sqrt(2. * pi) * sigma) - ((x - mu) * (x - mu)) / (2.0 * sigma * sigma)

def norm_lpdf_multivariate(x, mu, sigma):
	# Via http://stackoverflow.com/questions/11615664/multivariate-normal-density-in-python
	size = len(x)
	
	# some checks:
	if size != len(mu) or (size, size) != sigma.shape: raise NameError("The dimensions of the input don't match")	
	det = np.linalg.det(sigma)
	if det == 0: raise NameError("The covariance matrix can't be singular")

	norm_const = - size*log(2.0*pi)/2.0 - log(det)/2.0
	#norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
	x_mu = np.matrix(x - mu)
	inv = np.linalg.inv(sigma)        
	result = -0.5 * (x_mu * inv * x_mu.T)
	return norm_const + result

def logrange(mn,mx,steps):
	"""
		Logarithmically-spaced steps from mn to mx, with steps number inbetween
		mn - min value
		mx - max value
		steps - number of steps between. When 1, only mx is returned
	"""
	mn = np.log(mn)
	mx = np.log(mx)
	r = np.arange(mn, mx, (mx-mn)/(steps-1))
	r = np.append(r, mx)
	return np.exp(r)

def geometric_ldensity(n,p): 
	""" Log density of a geomtric distribution """
	return (n-1)*log(1.0-p)+log(p)

from math import expm1, log1p
def log1mexp(a):
	"""
		Computes log(1-exp(a)) according to Machler, "Accurately computing ..."
		Note: a should be a large negative value!
	"""
	if a > 0: print >>sys.stderr, "# Warning, log1mexp with a=", a, " > 0" 
	if a < -log(2.0): return log1p(-exp(a))
	else:             return log(-expm1(a))
	
	
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Sampling functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def sample1(*args): return sample_one(*args)
def sample_one(*args): 
	if len(args) == 1: return sample(args[0],1)[0] # use the list you were given
	else:             return sample(args, 1)[0]   # treat the arguments as a list

def flip(p): return (random() < p)


## TODO: THIS FUNCTION SUCKS PLEASE FIX IT
## TODO: Change this so that if N is large enough, you sort 
# takes unnormalized probabilities and returns a list of the log probability and the object
# returnlist makes the return always a list (even if N=1); otherwise it is a list for N>1 only
# NOTE: This now can take probs as a function, which is then mapped!
def weighted_sample(objs, N=1, probs=None, log=False, return_probability=False, returnlist=False, Z=None):
	"""
	
		when we return_probability, it is *always* a log probability
	"""
	# check how probabilities are specified
	# either as an argument, or attribute of objs (either probability or lp
	# NOTE: THis ALWAYS returns a log probability
	
	if len(objs) == 0: return None
	
	# convert to support indexing if we need it
	if isinstance(objs, set): 
		objs = list(objs) 
	
	myprobs = None
	if probs is None: # defaultly, we use .lp
		myprobs = [1.0] * len(objs) # sample uniform
	elif isinstance(probs, types.FunctionType): #NOTE: this does not work for class instance methods
		myprobs = map(probs, objs)
	else: 
		myprobs = map(float, probs)
	
	# Now normalize and run
	if Z == None:
		if log: Z = logsumexp(myprobs)
		else: Z = sum(myprobs)
	#print log, myprobs, Z
	out = []
	
	for n in range(N):
		r = random()
		for i in range(len(objs)):
			if log: r = r - exp(myprobs[i] - Z) # log domain
			else: r = r - (myprobs[i]/Z) # probability domain
			#print r, myprobs
			if r <= 0: 
				if return_probability: 
					lp = 0
					if log: lp = myprobs[i] - Z
					else:   lp = math.log(myprobs[i]) - math.log(Z)
				
					out.append( [objs[i],lp] )
					break
				else:             
					out.append( objs[i] )
					break
					
	if N == 1 and (not returnlist): return out[0]  #don't give back a list if you just want one
	else:      return out

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Lambda calculus
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Some innate lambdas
def lambdaZero(*x): return 0
def lambdaOne(*x): return 1
def lambdaNull(*x): return []
def lambdaNone(*x): return None
def lambdaTrue(*x): return True
def lambdaFalse(*x): return True
def lambdaNAN(*x): return float("nan")

"""
The Y combinator
#example:
#fac = lambda f: lambda n: (1 if n<2 else n*(f(n-1)))
#Y(fac)(10)
"""
Y = lambda f: (lambda x: x(x)) (lambda y : f(lambda *args: y(y)(*args)) )


class RecursionDepthException(Exception):
	# An exception class for recursing too deep
	def __init__(self): pass


MAX_RECURSION = 25
def Y_bounded(f):
	"""
	A fancy fixed point iterator that only goes MAX_RECURSION deep, else throwing a a RecusionDepthException
	"""
	return (lambda x, n: x(x, n)) (lambda y, n: f(lambda *args: y(y, n+1)(*args)) if n < MAX_RECURSION else raise_exception(RecursionDepthException()), 0)




def Ystar(*l):
	"""
	The Y* combinator for mutually recursive functions. Holy shit.
	
	(define (Y* . l)
          ((lambda (u) (u u))
            (lambda (p) (map (lambda (li) (lambda x (apply (apply li (p p)) x))) l))))
	
	http://okmij.org/ftp/Computation/fixed-point-combinators.html]
	
	http://stackoverflow.com/questions/4899113/fixed-point-combinator-for-mutually-recursive-functions
	
	
	Here is even/odd:
	
	
	even,odd = Ystar( lambda e,o: lambda x: (x==0) or o(x-1), \
                          lambda e,o: lambda x: (not x==0) and e(x-1) )
                          
        Note that we require a lambda e,o on the outside so that these can have names inside.
	"""
	
	return (lambda u: u(u))(lambda p: map(lambda li: lambda x: apply(li, p(p))(x), l))



"""
	Evaluation of expressions
"""

def evaluate_expression(e, args=['x'], recurse="L_", addlambda=True):
	"""
	This evaluates an expression. If 
	- e         - the expression itself -- either a str or something that can be made a str
	- addlambda - should we include wrapping lambda arguments in args? lambda x: ...
	- recurse   - if addlambda, this is a special primitive name for recursion
	- args      - if addlambda, a list of all arguments to be added
	
	g = evaluate_expression("x*L(x-1) if x > 1 else 1")
	g(12)
	"""
	
	if not isinstance(e,str): e = str(e)
	f = None # the function
	
	try:
		if addlambda:
			f = eval('lambda ' + recurse + ': lambda ' + ','.join(args) + ' :' + e)
			return Y_bounded(f)
		else: 
			f = eval(e)
			return f
	except:
		print "Error in evaluate_expression:", e
		raise RuntimeError
		exit(1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  And import the primitives for "eval"
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.BasicPrimitives import * # Needed for calling "eval"

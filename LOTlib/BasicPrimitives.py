# -*- coding: utf-8 -*-


"""
	Primitives that may be used in the LOT

	
	TODO: Include "multiset" objects so that union can either take into account uniqueness or not!
	
"""
from LOTlib.Miscellaneous import *
from LOTlib.FunctionNode import isFunctionNode
import re

import math

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We define two variables, one for how many function calls have been
# used in a single function/hypothesis, and one for how many have been
# run over the entire course of the experiment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LOCAL_PRIMITIVE_OPS = 0
GLOBAL_PRIMITIVE_OPS = 0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A decorator for basic primitives that increments our counters 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def LOTlib_primitive(fn):
	def inside(*args, **kwargs):
		
		global LOCAL_PRIMITIVE_OPS 
		LOCAL_PRIMITIVE_OPS += 1
		
		global GLOBAL_PRIMITIVE_OPS 
		GLOBAL_PRIMITIVE_OPS += 1
		
		#if None2None and any([a is None for a in args]): return False
		
		return fn(*args, **kwargs)
		
	return inside
	
def None2None(fn):
	"""
		A decorator to map anything with "None" as a *list* arg (NOT a keyword arg)
		this will make it return None overall
		
		If you want to have this not prevent incrementing (via LOTlib_primitive), then 
		we need to put it *after* LOTlib_primitive:
		
		@LOTlib_primitive
		@None2None
		def f(...):
	"""
	
	def inside(*args, **kwargs): 
		if any([a is None for a in args]): return None
		return fn(*args, **kwargs)
		
	return inside
	
	
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Constants
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	
		
PI = math.pi
TAU = 2.0*PI
E = math.e

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Lambda calculus & Scheme
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@LOTlib_primitive
@None2None
def apply_(f,x):
	return f(x)
	
@LOTlib_primitive
@None2None
def cons_(x,y):
	try: return [x,y]
	except: return None

@LOTlib_primitive
@None2None
def cdr_(x):
	try: return x[1:]
	except: return None

rest_  = cdr_

@LOTlib_primitive
@None2None
def car_(x):
	try: return x[0]
	except: return None

first_ = car_

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Combinators -- all curried
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@LOTlib_primitive
@None2None
def I_(x):
	return x
	
@LOTlib_primitive
@None2None
def K_(x): # constant function
	return (lambda y: x)
	
@LOTlib_primitive	
@None2None
def S_(x): #(S x y z) = (x z (y z))
	# (S x) --> lambda y lambda z: 
	return lambda y: lambda z: x(z)( y(z) )
	
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For language / semantics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@LOTlib_primitive
@None2None
def presup_(a,b):
	if a: return b
	else: 
		if b: return "undefT" # distinguish these so that we can get presup out 
		else: return "undefF"
		
@LOTlib_primitive
@None2None
def is_undef(x):
	if isinstance(x,list): 
		return map(is_undef, x)
	else:
		return (x is None) or (x =="undefT") or (x == "undefF") or (x == "undef")
		
@LOTlib_primitive
@None2None
def collapse_undef(x):
	"""
		Change undefT->True and undefF->False
	"""
	if isinstance(x,list): return map(collapse_undef, x)
	else:  
		if    x is "undefT": return True
		elif  x is "undefF": return False
		else: x

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Assembly arithmetic
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def assembly_primitive(fn):
	def inside(*args):
		try:
			v = fn(*args)
			return v
		except ZeroDivisionError: return float("nan")
		except ValueError: return float("nan")
		except OverflowError: return float("nan")
		
	return inside



# For assembly 
@assembly_primitive
def ADD(x,y): return x+y

@assembly_primitive
def SUB(x,y): return x-y

@assembly_primitive
def MUL(x,y): return x*y

@assembly_primitive
def DIV(x,y): return x/y

@assembly_primitive
def LOG(x): return log(x)

@assembly_primitive
def POW(x,y):return pow(x,y)

@assembly_primitive
def EXP(x): return exp(x)

@assembly_primitive
def NEG(x): return -x

@assembly_primitive
def SIN(x): return math.sin(x)

@assembly_primitive
def ASIN(x): return math.asin(x)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic arithmetic
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@LOTlib_primitive
@None2None
def negative_(x): return -x

@LOTlib_primitive
@None2None
def plus_(x,y): return x+y

@LOTlib_primitive
@None2None
def times_(x,y): return x*y

@LOTlib_primitive
@None2None
def divide_(x,y): 
	if y > 0: return x/y
	else:     return float("inf")*x
	
@LOTlib_primitive
@None2None
def subtract_(x,y): return x-y

	
@LOTlib_primitive
@None2None
def minus_(x,y): return x-y

@LOTlib_primitive
@None2None
def sin_(x): 
	try:
		return math.sin(x)
	except: return float("nan")

@LOTlib_primitive
@None2None
def cos_(x): 
	try:
		return math.cos(x)
	except: return float("nan")

@LOTlib_primitive
@None2None
def tan_(x): 
	try:
		return math.tan(x)
	except: return float("nan")

@LOTlib_primitive	
@None2None
def sqrt_(x): 
	try: return math.sqrt(x)
	except: return float("nan")

@LOTlib_primitive
@None2None
def pow_(x,y): 
	#print x,y
	try: return pow(x,y)
	except: return float("nan")
	
@LOTlib_primitive
@None2None
def exp_(x): 
	try: 
		r = math.exp(x)
		return r
	except: 
		return float("inf")*x

@LOTlib_primitive
@None2None
def abs_(x): 
	try: 
		r = abs(x)
		return r
	except: 
		return float("inf")*x
		
		
@LOTlib_primitive	
@None2None
def log_(x): 
	if x > 0: return math.log(x)
	else: return -float("inf")

@LOTlib_primitive
@None2None
def log2_(x): 
	if x > 0: return math.log(x)/log(2.0)
	else: return -float("inf")
	
@LOTlib_primitive
@None2None
def pow2_(x): 
	return math.pow(2.0,x)

@LOTlib_primitive
@None2None
def mod_(x,y): return (x%y)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic logic
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
https://en.wikipedia.org/wiki/Truth_function#Table_of_binary_truth_functions
	
"""


@LOTlib_primitive
@None2None
def id_(A): return A # an identity function

@LOTlib_primitive
@None2None
def and_(A,B): return (A and B)

@LOTlib_primitive
@None2None
def AandnotB_(A,B): return (A and (not B))

@LOTlib_primitive
@None2None
def notAandB_(A,B): return ((not A) and B)

@LOTlib_primitive
@None2None
def AornotB_(A,B): return (A or (not B))

@LOTlib_primitive
@None2None
def A_(A,B): return A

@LOTlib_primitive
@None2None
def notA_(A,B): return not A

@LOTlib_primitive
@None2None
def B_(A,B): return B

@LOTlib_primitive
@None2None
def notB_(A,B): return not B

@LOTlib_primitive
@None2None
def nand_(A,B): return not (A and B)

@LOTlib_primitive
@None2None
def or_(A,B): return (A or B)

@LOTlib_primitive
@None2None
def nor_(A,B): return not (A or B)

@LOTlib_primitive
@None2None
def xor_(A,B): return (A and (not B)) or ((not A) and B)

@LOTlib_primitive
@None2None
def not_(A): return (not A)

@LOTlib_primitive
@None2None
def implies_(A,B): return (A or (not B))

@LOTlib_primitive
@None2None
def iff_(A,B): return ((A and B) or ((not A) and (not B)))

@LOTlib_primitive
@None2None
def if_(C,X,Y):
	if C: return X
	else: return Y

@LOTlib_primitive
@None2None
def ifU_(C,X):
	if C: return X
	else: return 'undef'

@LOTlib_primitive
@None2None
def gt_(x,y): return x>y

@LOTlib_primitive
@None2None
def gte_(x,y): return x>=y

@LOTlib_primitive
@None2None
def lt_(x,y): return x<y

@LOTlib_primitive
@None2None
def lte_(x,y): return x<=y

@LOTlib_primitive
@None2None
def eq_(x,y): return x==y

@LOTlib_primitive
@None2None
def zero_(x,y): return x==0.0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set-theoretic primitives
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@LOTlib_primitive
@None2None
def union_(A,B): return A.union(B)

@LOTlib_primitive
@None2None
def intersection_(A,B): return A.intersection(B)

@LOTlib_primitive
@None2None
def setdifference_(A,B): return A.difference(B)

@LOTlib_primitive
@None2None
def select_(A): # choose an element, but don't remove it
	if len(A) > 0:
		x = A.pop()
		A.add(x)
		return set([x]) # but return a set
	else: return set() # empty set

@LOTlib_primitive
@None2None
def exhaustive_(A,B): return coextensive(A,B)

@LOTlib_primitive
@None2None
def coextensive_(A,B): return coextensive(A,B)
def coextensive(A,B): # are the two sets coextensive?
	#print A,B
	return (A.issubset(B) and B.issubset(A))

@LOTlib_primitive
@None2None
def equal_(A,B): return (A == B)

@LOTlib_primitive
@None2None
def equal_word_(A,B): return (A == B)

@LOTlib_primitive
@None2None
def empty_(A): return (len(A)==0)

@LOTlib_primitive
@None2None
def nonempty_(A): return not empty_(A)

@LOTlib_primitive
@None2None
def cardinality1_(A): return (len(A)==1)

@LOTlib_primitive
@None2None
def cardinality2_(A): return (len(A)==2)

@LOTlib_primitive
@None2None
def cardinality3_(A): return (len(A)==3)

@LOTlib_primitive
@None2None
def cardinality4_(A): return (len(A)==4)

@LOTlib_primitive
@None2None
def cardinality5_(A): return (len(A)==5)

@LOTlib_primitive
@None2None
def cardinality_(A): return len(A)

# returns cardinalities of sets and otherwise numbers -- for duck typing sets/ints
def cardify(x):
	if isinstance(x, set): return len(x)
	else: return x

@LOTlib_primitive
@None2None
def cardinalityeq_(A,B): return cardify(A) == cardify(B)

@LOTlib_primitive
@None2None
def cardinalitygt_(A,B): return cardify(A) > cardify(B)

@LOTlib_primitive
@None2None
def cardinalitylt_(A,B): return cardify(A) > cardify(B)

@LOTlib_primitive
@None2None
def subset_(A,B):
	return A.issubset(B)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Quantification
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@LOTlib_primitive
@None2None
def not_exists_(F,S): return not exists_(F,S)

@LOTlib_primitive
@None2None
def exists_(F,S): return exists(F,S)
def exists(F,S):
	#if not isinstance(S,list): return None
	for s in S:
		if F(s): return True
	return False

@LOTlib_primitive	
@None2None
def not_forall_(F,S): return not forall(F,S)	

@LOTlib_primitive
@None2None
def forall_(F,S): return forall(F,S)

def forall(F,S):
	#if not isinstance(S,list): return None
	for s in S:
		if not F(s): return False
	return True
	
@LOTlib_primitive
@None2None
def iota_(F,S):
	"""
		The unique F in S. If none, or not unique, return None
	"""
	match = None
	for s in S:
		if F(s):
			if match is None: match = s
			else: return None  # We matched more than one
	return match
		
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Tree operations
# In a tree T, check relations between some elements. Sometimes T is 
# not used, but we leave it in all functions for simplicity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@LOTlib_primitive
@None2None
def is_(x,y): return (x is y)

@LOTlib_primitive
@None2None
def co_referents_(T,x):
	"""
		The co-referents of x in t
	"""
	return filter(lambda si: co_refers(si,x), T)

@LOTlib_primitive
@None2None
def sisters_(T, x, y, equality=False):
	"""
		Check if x,y are sisters in T
	"""
	for s in T:
		if immediately_dominates(s,x) and immediately_dominates(s,y): return True
	return False

#@ We define a non-LOTlib version so we can use this in other definitions
def immediately_dominates(x, y):
	if isinstance(x, FunctionNode):
		for s in x.args:
			if s is y: return True
	
	return False
	
@LOTlib_primitive
@None2None
def immediately_dominates_(x, y):
	return immediately_dominates(x,y)
	
@LOTlib_primitive
@None2None
def dominates_(x,y):
	"""
		This checks if x >> y, but using object identity ("is") rather than equality
	"""
	if x is y: return False # A node does not dominate itself
	if isinstance(x, FunctionNode):
		for s in x:
			if s is y: return True
	return False
	
@LOTlib_primitive
@None2None
def tree_up_(T,x): return tree_up(T,x)

#@Define a non-LOTlib version for defining others
def tree_up(T, x):
	"""
		Go up one node in the tree. If you are root, return None
		
		NOTE: Super slow since we search over the whole tree each time.... This is a little tricky otherwise without pointers
	"""
	
	if x is T: return None
	
	for s in T:
		if immediately_dominates(s,x): return s
	return None

	
@LOTlib_primitive
@None2None
def children_(x): 
	if isinstance(x, FunctionNode): return [ c for c in x.args ]
	else: return []
	
@LOTlib_primitive	
@None2None
def descendants_(x):        
	if isinstance(x, FunctionNode): return [ c for c in x ]
	else: return []

@LOTlib_primitive
@None2None
def tree_root_(T):
	return T

@LOTlib_primitive
@None2None
def is_nonterminal_type_(x,y): return is_nonterminal_type(x,y)

no_coref_regex = re.compile(r"\..+$")
def is_nonterminal_type(x,y):
	# Check if x is of a given type, but remove corefence information from X (y is the type)
	
	if x is None or y is None: return False
	
	if not isinstance(x,str): x = x.name
	
	# remove the .coreference info
	x = no_coref_regex.sub("", x)
	
	return (x==y)

@LOTlib_primitive
@None2None
def ancestors_(T, x): return ancestors(T,x)

#def ancestors(T,x):
	### SLOW VERSION -- O(N^2) since tree_up is O(N)
	#if not isinstance(x, FunctionNode): return []
	#out = []
	#while not tree_is_(x,T):
		#x = tree_up(T,x)
		#out.append(x)
	#return out

def ancestors(T,x):
	"""
		Here is a version of ancestors that is O(n), rather than the repeated calls to tree_up, which is O(N^2)
	"""
	
	anc = []
	
	def recurse_down(y):
		#print "RD", y, "\t", x
		if isinstance(y,list):
			return any(map(recurse_down, filter(isFunctionNode, y)))
		elif isFunctionNode(y):
			if recurse_down(y.args) or immediately_dominates(y, x): 
				anc.append(y) # put y on the end
				return True
			return False
	
	recurse_down(T)
	
	return anc
	
@LOTlib_primitive
@None2None
def whole_tree_(T):
	# LIST type of all elements of T
	return [ti for ti in T ]

@LOTlib_primitive
@None2None
def tree_is_(x,y): return (x is y)

## Co-reference (via strings)
coref_matcher = re.compile(r".+\.([0-9]+)$")
@LOTlib_primitive
@None2None
def co_refers_(x,y): return co_refers(x,y)

def co_refers(x,y):
	
	if x is y: return False # By stipulation, nothing co-refers to itself
	
	## Check if two FunctionNodes or strings co-refer (e.g. are indexed with the same .i in their name)
	xx = x.name if isFunctionNode(x) else x
	yy = y.name if isFunctionNode(y) else y
	
	mx = coref_matcher.search(xx) 
	my = coref_matcher.search(yy)
		
	if mx is None or my is None: 
		return False
	else: 
		return (mx.groups("X")[0] == my.groups("Y")[0]) # set the default in groups so that they won't be equal if there is no match

@LOTlib_primitive
@None2None
def non_xes_(x,T):
	return filter(lambda v: v is not x, T)

@LOTlib_primitive
@None2None
def first_dominating_(T,x,t):
	# Returns the first thing dominating x of type t
	# And None otherwise
	
	if isFunctionNode(x):
		up = tree_up(T,x)
		while up is not None:
			if is_nonterminal_type(up,t): return up
			up = tree_up(T,up)
	
	return None

@LOTlib_primitive
@None2None
def first_dominated_(x,t):
	# Returns the first thing dominating x of type t
	# And None otherwise
	if isFunctionNode(x):
		for sn in x:
			if is_nonterminal_type(sn, t): return sn
		
	return None
	
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# closure operations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## TODO: Add transitive closure of an operation
@LOTlib_primitive
@None2None
def filter_(f,x):
	return filter(f,x)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# counting list
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from collections import defaultdict

# the next word in the list -- we'll implement these as a hash table
word_list = ['one_', 'two_', 'three_', 'four_', 'five_', 'six_', 'seven_', 'eight_', 'nine_', 'ten_']
next_hash, prev_hash = [defaultdict(lambda: 'undef'), defaultdict(lambda: 'undef')]
for i in range(1, len(word_list)-1):
	next_hash[word_list[i]] = word_list[i+1]
	prev_hash[word_list[i]] = word_list[i-1]
next_hash['one_'] = 'two_'
next_hash['ten_'] = 'undef'
prev_hash['one_'] = 'undef'
prev_hash['ten_'] = 'nine_'
next_hash['X'] = 'X'
prev_hash['X'] = 'X'

word_to_number = dict() # map a word like 'four_' to its number, 4	
for i in range(len(word_list)):
	word_to_number[word_list[i]] = i+1
word_to_number['ten_'] = 'A' # so everything is one character

prev_hash[None] = None

@LOTlib_primitive
@None2None
def next_(w): return next_hash[w]

@LOTlib_primitive
@None2None
def prev_(w): return prev_hash[w]

# and define these
one_ = 'one_'
two_ = 'two_'
three_ = 'three_'
four_ = 'four_'
five_ = 'five_'
six_ = 'six_'
seven_ = 'seven_'
eight_ = 'eight_'
nine_ = 'nine_'
ten_ = 'ten_'
undef = 'undef'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Access arbitrary features
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@LOTlib_primitive
def F1(x): return x.F1

@LOTlib_primitive
def F2(x): return x.F2

@LOTlib_primitive
def F3(x): return x.F3

@LOTlib_primitive
def F4(x): return x.F4

@LOTlib_primitive
def F5(x): return x.F5

@LOTlib_primitive
def F6(x): return x.F6

@LOTlib_primitive
def F7(x): return x.F7

@LOTlib_primitive
def F8(x): return x.F8

@LOTlib_primitive
def F9(x): return x.F9

@LOTlib_primitive
def F10(x): return x.F10

# Some of our own primitivesS
@LOTlib_primitive
@None2None
def is_color_(x,y): return (x.color == y)

@LOTlib_primitive
@None2None
def is_shape_(x,y): return (x.shape == y)

@LOTlib_primitive
@None2None
def is_pattern_(x,y): return (x.pattern == y)


from LOTlib.FunctionNode  import FunctionNode
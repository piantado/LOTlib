# -*- coding: utf-8 -*-


"""
	Primitives that may be used in the LOT

	
	TODO: Include "multiset" objects so that union can either take into account uniqueness or not!
	
"""
from LOTlib.Miscellaneous import *

import re

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Lambda calculus & Scheme
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def apply_(f,x):
	return f(x)
	
def cons_(x,y):
	try: return [x,y]
	except: return None

def cdr_(x):
	try: return x[1:]
	except: return None

def car_(x):
	try: return x[0]
	except: return None

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Combinators -- all curried
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def I(x):
	return x

def K(x): # constant function
	return (lambda y: x)
	
def S(x): #(S x y z) = (x z (y z))
	# (S x) --> lambda y lambda z: 
	return lambda y: lambda z: x(z)( y(z) )
	
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For language / semantics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def presup_(a,b):
	if a: return b
	else: 
		if b: return "undefT" # distinguish these so that we can get presup out 
		else: return "undefF"

def is_undef(x):
	if isinstance(x,list): 
		return map(is_undef, x)
	else:
		return (x is None) or (x =="undefT") or (x == "undefF") or (x == "undef")

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
# Basic arithmetic
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import math

def negative_(x): return -x

def plus_(x,y): return x+y
def times_(x,y): return x*y
def divide_(x,y): 
	if y > 0: return x/y
	else:     return float("inf")*x
def subtract_(x,y): return x-y

def sin_(x): 
	try:
		return math.sin(x)
	except: return float("nan")
def cos_(x): 
	try:
		return math.cos(x)
	except: return float("nan")
def tan_(x): 
	try:
		return math.tan(x)
	except: return float("nan")
	
def sqrt_(x): 
	try: return math.sqrt(x)
	except: return float("nan")
	
def pow_(x,y): 
	#print x,y
	try: return pow(x,y)
	except: return float("nan")

def exp_(x): 
	try: 
		r =math.exp(x)
		return x
	except: 
		return float("inf")*x
		
def log_(x): 
	if x > 0: return math.log(x)
	else: return -float("inf")

def log2_(x): 
	if x > 0: return math.log(x)/log(2.0)
	else: return -float("inf")
def pow2_(x): 
	return math.pow(2.0,x)

def mod_(x,y): return (x%y)

PI = math.pi
E = math.e

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic logic
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def id_(A): return A # an identity function

def and_(A,B): return (A and B)
def nand_(A,B): return not (A and B)
def or_(A,B): return (A or B)
def nor_(A,B): return not (A or B)
def xor_(A,B): return (A and (not B)) or ((not A) and B)
def not_(A): return (not A)
def implies_(A,B): return (A or (not B))
def iff_(A,B): return ((A and B) or ((not A) and (not B)))

def if_(C,X,Y):
	if C: return X
	else: return Y
def ifU_(C,X):
	if C: return X
	else: return 'undef'
	
def gt_(x,y): return x>y
def gte_(x,y): return x>=y
def lt_(x,y): return x<y
def lte_(x,y): return x<=y
def eq_(x,y): return x==y
def zero_(x,y): return x==0.0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set-theoretic primitives
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def union_(A,B): return A.union(B)
def intersection_(A,B): return A.intersection(B)
def setdifference_(A,B): return A.difference(B)
def select_(A): # choose an element, but don't remove it
	if len(A) > 0:
		x = A.pop()
		A.add(x)
		return set([x]) # but return a set
	else: return set() # empty set


def exhaustive_(A,B): return coextensive_(A,B)
def coextensive_(A,B): # are the two sets coextensive?
	#print A,B
	return (A.issubset(B) and B.issubset(A))

def equal_(A,B): return (A == B)
def equal_word_(A,B): return (A == B)

def empty_(A): return (len(A)==0)
def nonempty_(A): return not empty_(A)
def cardinality1_(A): return (len(A)==1)
def cardinality2_(A): return (len(A)==2)
def cardinality3_(A): return (len(A)==3)
def cardinality4_(A): return (len(A)==4)
def cardinality5_(A): return (len(A)==5)
def cardinality_(A): return len(A)

# returns cardinalities of sets and otherwise numbers
def cardify(x):
	if isinstance(x, set): return len(x)
	else: return x

def cardinalityeq_(A,B): return cardify(A) == cardify(B)
def cardinalitygt_(A,B): return cardify(A) > cardify(B)
def cardinalitylt_(A,B): return cardify(A) > cardify(B)

def subset_(A,B):
	return A.issubset(B)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Quantification
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def not_exists_(F,S): return not exists_(F,S)
def exists_(F,S):
	#if not isinstance(S,list): return None
	for s in S:
		if F(s): return True
	return False
	
def not_forall_(F,S): return not forall_(F,S)	
def forall_(F,S):
	#if not isinstance(S,list): return None
	for s in S:
		if not F(s): return False
	return True

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

def sisters_(T, x, y, equality=False):
	"""
		Check if x,y are sisters in T
	"""
	for s in T.all_subnodes():
		if immediately_dominates_(s,x) and immediately_dominates_(s,y): return True
	return False

def immediately_dominates_(x, y):
	if isinstance(x, FunctionNode):
		for s in x.args:
			if s is y: return True
	
	return False
	
def dominates_(x,y):
	"""
		This checks if x >> y, but using object identity ("is") rather than equality
	"""
	if x is y: return False # A node does not dominate itself
	if isinstance(x, FunctionNode):
		for s in x.all_subnodes():
			if s is y: return True
	return False

def tree_up_(T, x):
	"""
		Go up one node in the tree
		BUT if you are the root (T), then return yourself
	"""
	
	if x is T: return T
	
	for s in T.all_subnodes():
		if immediately_dominates_(s,x): return s
	return None

def children_(x): 
	if isinstance(x, FunctionNode): return [ c for c in x.args ]
	else: return []
	
def descendants_(x):        
	if isinstance(x, FunctionNode): return [ c for c in x.all_subnodes() ]
	else: return []
	
def tree_root_(T):
	return T
	
def is_nonterminal_type_(x,y):
	# Check if x is of a given type, but remove corefence information from X (y is the type)
	
	if x is None or y is None: return False
	
	if not isinstance(x,str): x = x.name
	
	# remove the .coreference info
	x = re.sub(r"\..+$", "", x)
	
	return (x==y)
		
def ancestors_(T, x):
	if not isinstance(x, FunctionNode): return []
	out = []
	while not tree_is_(x,T):
		x = tree_up_(T,x)
		out.append(x)
	return out
	
def whole_tree_(T):
	# LIST type of all elements of T
	return [ti for ti in T.all_subnodes() ]
	
def tree_is_(x,y): return (x is y)

## Co-reference (via strings)
coref_matcher = re.compile(r".+\.([0-9]+)$")
def co_refers_(x,y):
	
	if x is y: return True # Hmm should have this, I Think (regardless of type, etc)
	
	## Check if two FunctionNodes or strings co-refer (e.g. are indexed with the same .i in their name)
	xx = x.name if isinstance(x, FunctionNode) else x
	yy = y.name if isinstance(y, FunctionNode) else y
	
	mx = coref_matcher.search(xx) 
	my = coref_matcher.search(yy)
	
	#print ">>>", x
	#print ">>>", y
	#if mx is None or my is None: print "--", "FALSE"
	#else:                        print "--", mx.groups("X")[0], my.groups("Y")[0], (mx.groups("X")[0] == my.groups("Y")[0])
		
	if mx is None or my is None: 
		return False
	else: 
		return (mx.groups("X")[0] == my.groups("Y")[0]) # set the default in groups so that they won't be equal if there is no match

def non_xes_(x,T):
	#print ">>", T
	return filter(lambda v: v is not x, T)
		
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# closure operations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## TODO: Add transitive closure of an operation

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
def next_(w): return next_hash[w]
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
def F1(x): return x.F1
def F2(x): return x.F2
def F3(x): return x.F3
def F4(x): return x.F4
def F5(x): return x.F5
def F6(x): return x.F6
def F7(x): return x.F7
def F8(x): return x.F8
def F9(x): return x.F9
def F10(x): return x.F10

# Some of our own primitivesS
def is_color_(x,y): return (x.color == y)
def is_shape_(x,y): return (x.shape == y)
def is_pattern_(x,y): return (x.pattern == y)


from LOTlib.FunctionNode  import FunctionNode
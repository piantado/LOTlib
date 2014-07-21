
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import evaluate_expression

"""
	 A simple demo for how to define FunctionNodes
	 
	 All of these primitives are defined in the LOTlib.Primitives package, as well as many others.
	 In general, the PCFG generates FunctionNode trees via "generate" and then these are printed
	 via __str__ in a pythonesque way that can be evaled. 

"""

grammar = Grammar()

# Nonterminal START -> Nonterminal EXPR (with no function call)
grammar.add_rule('START', '', ['EXPR'], 1.0) 

# And "EXPR" can rewrite as "1.0" -- and this expansion has probability proportional to 5.0
grammar.add_rule('EXPR', '1.0', None, 5.0) 

# some other simple terminals
# these are given much higher probability in order to keep the PCFG well-defined
grammar.add_rule('EXPR', '0.0', None, 3.0) 
grammar.add_rule('EXPR', 'pi', None, 3.0) 
grammar.add_rule('EXPR', 'e', None, 3.0) 

# To have a string terminal, it must be quoted:
#grammar.add_rule('EXPR', '\'e\'', None, 3.0) 

# Then this is one way to use the variable "x" of the function. 
# This gets named as the argument in evaluate_expression below
grammar.add_rule('EXPR', 'x', None, 10.0) 

# A thunk function (lambdaZero is defined in Miscellaneous)
# We write these with [None] insead of []. The FunctionNode str function knows to print these with parens
# This notation keeps it simple since on a FunctionNode, the children ("to") are always a list. 
grammar.add_rule('EXPR', 'lambdaZero', [], 1.0) 
#or
grammar.add_rule('EXPR', 'lambdaZero()', None, 1.0)  # this is supported but not recommended


# EXPR -> plus_(EXPR, EXPR)
grammar.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], 1.0)

# Or other operations
grammar.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], 1.0)
grammar.add_rule('EXPR', 'subtract_', ['EXPR', 'EXPR'], 1.0)
grammar.add_rule('EXPR', 'divide_', ['EXPR', 'EXPR'], 1.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Lambda expressions:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# EXPR -> apply(FUNCTION, EXPR)
grammar.add_rule('EXPR', 'apply_', ['FUNCTION', 'EXPR'], 5.0)

# Here, 'lambda' is a special function that allows us to introduce a new bound variable (bv) of a cetain type.
# The type is specified by bv_args:

grammar.add_rule('FUNCTION', 'lambda', ['EXPR'], 1.0,  bv_name=None, bv_args=None) # Creates a thunk -- no variables, but gets evaled like a lambda (does not add rules)

grammar.add_rule('FUNCTION', 'lambda', ['EXPR'], 1.0,  bv_name='BOOL', bv_args=None) # Creates a terminal of type bool -- e.g. y1

grammar.add_rule('FUNCTION', 'lambda', ['EXPR'], 1.0,  bv_name='BOOL', bv_args=[]) # Creates a thunk lambda variable -- e.g y1()

grammar.add_rule('FUNCTION', 'lambda', ['EXPR'], 1.0,  bv_name='BOOL', bv_args=['EXPR']) # Creates a lambda variable yi always called with an EXPR argument -- e.g., y1(plus(1,1))

# Etc. 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Conditional:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# if_ gets printed specially (see LOTlib.FunctionNode.__str__). Here COND is a name that is made up 
# here for conditional expressions 
grammar.add_rule('EXPR', 'if_', ['COND', 'EXPR', 'EXPR'], 1.0)
grammar.add_rule('COND', 'gt_', ['EXPR', 'EXPR'], 1.0)
grammar.add_rule('COND', 'eq_', ['EXPR', 'EXPR'], 1.0)
# Note that because if_ prints specially, it is correctly handled (via short circuit evaluation)
# so that we don't eval both branches unnecessarily


for _ in xrange(1000):
	
	t = grammar.generate() # Default is to generate from 'START'; else use 'START=t' to generate from type t
	
	# Now x is a FunctionNode
	
	# We can compile it via LOTlib.Miscellaneous.evaluate_expression
	# This says that t is a *function* with arguments 'x' (allowed via the grammar above)
	# The alternative way to do this would be to put a lambda at the top of each tree
	f = evaluate_expression(t, args=['x'])
	
	print t # will call x.__str__ and display as a pythonesque string
	print map(f, range(0,10))








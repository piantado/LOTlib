
from LOTlib.PCFG import PCFG
from LOTlib.Miscellaneous import evaluate_expression

"""
	 A simple demo for how to define FunctionNodes
	 
	 All of these primitives are defined in LOTlib.BasicPrimitives, as well as many others.
	 In general, the PCFG generates FunctionNode trees via "generate" and then these are printed
	 via __str__ in a pythonesque way that can be evaled. 

"""

G = PCFG()

# Nonterminal START -> Nonterminal EXPR (with no function call)
G.add_rule('START', '', ['EXPR'], 1.0) 

# And "EXPR" can rewrite as "1.0" -- and this expansion has probability proportional to 5.0
G.add_rule('EXPR', '1.0', [], 5.0) 

# some other simple terminals
# these are given much higher probability in order to keep the PCFG well-defined
G.add_rule('EXPR', '0.0', [], 3.0) 
G.add_rule('EXPR', 'pi', [], 3.0) 
G.add_rule('EXPR', 'e', [], 3.0) 

# Then this is one way to use the variable "x" of the function. 
# This gets named as the argument in evaluate_expression below
G.add_rule('EXPR', 'x', [], 10.0) 

# A thunk function (lambdaZero is defined in BasicPrimitives)
# We write these with [None] insead of []. The FunctionNode str function knows to print these with parens
# This notation keeps it simple since on a FunctionNode, the children ("to") are always a list. 
G.add_rule('EXPR', 'lambdaZero', [None], 1.0)  

# EXPR -> plus_(EXPR, EXPR)
G.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], 1.0)

# Or other operations
G.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], 1.0)
G.add_rule('EXPR', 'subtract_', ['EXPR', 'EXPR'], 1.0)
G.add_rule('EXPR', 'divide_', ['EXPR', 'EXPR'], 1.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Lambda expressions:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# EXPR -> apply(FUNCTION, EXPR)
G.add_rule('EXPR', 'apply_', ['FUNCTION', 'EXPR'], 5.0)

# Here, 'lambda' is a special function that allows us to introduce a new bound variable (bv) of
# type EXPR (via bv='EXPR')
G.add_rule('FUNCTION', 'lambda', ['EXPR'], 1.0, bv=['EXPR'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Conditional:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# if_ gets printed specially (see LOTlib.FunctionNode.__str__). Here COND is a name I made up
# here for conditional expressions 
G.add_rule('EXPR', 'if_', ['COND', 'EXPR', 'EXPR'], 1.0)
G.add_rule('COND', 'gt_', ['EXPR', 'EXPR'], 1.0)
G.add_rule('COND', 'eq_', ['EXPR', 'EXPR'], 1.0)

for _ in xrange(1000):
	
	t = G.generate() # Default is to generate from 'START'; else use 'START=t' to generate from type t
	
	# Now x is a FunctionNode
	
	# We can compile it via LOTlib.Miscellaneous.evaluate_expression
	# This says that t is a *function* with arguments 'x' (allowed via the grammar above)
	# The alternative way to do this would be to put a lambda at the top of each tree
	f = evaluate_expression(t, args=['x'])
	
	print t # will call x.__str__ and display as a pythonesque string
	print map(f, range(0,10))








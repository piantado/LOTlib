
from LOTlib.Grammar import Grammar
from LOTlib.Evaluation.Eval import evaluate_expression, register_primitive


"""
     A simple demo for how to define FunctionNodes

     All of these primitives are defined in the LOTlib.Primitives package, as well as many others.
     In general, the PCFG generates FunctionNode trees via "generate" and then these are printed
     via str(...).
"""
grammar = Grammar()

# Nonterminal START -> Nonterminal EXPR (with no function call)
grammar.add_rule('START', '', ['EXPR'], 1.0)

# And "EXPR" can rewrite as "1.0" -- and this expansion has probability proportional to 5.0
grammar.add_rule('EXPR', '1.0', None, 5.0)

# some other simple terminals
# these are given much higher probability in order to keep the PCFG well-defined
grammar.add_rule('EXPR', '0.0', None, 3.0)
grammar.add_rule('EXPR', 'TAU', None, 3.0)
grammar.add_rule('EXPR', 'E', None, 3.0)

# To have a string terminal, it must be quoted:
#grammar.add_rule('EXPR', '\'e\'', None, 3.0)

# Then this is one way to use the variable "x" of the function.
# This gets named as the argument in evaluate_expression below
grammar.add_rule('EXPR', 'x', None, 10.0)

 # We can register a new function that will be evaled via evaluate_expression
def mylambda(): return 141.421
register_primitive(mylambda)

# A thunk function
grammar.add_rule('EXPR', 'mylambda', [], 1.0)
#or
grammar.add_rule('EXPR', 'mylambda()', None, 1.0)  # this is supported but not recommended
#grammar.add_rule('EXPR', 'mylambda', None, 1.0)  # this would have made mylambda as a non-function constant

# EXPR -> plus_(EXPR, EXPR)
grammar.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], 1.0)

# Or other operations
grammar.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], 1.0)
grammar.add_rule('EXPR', 'subtract_', ['EXPR', 'EXPR'], 1.0)
grammar.add_rule('EXPR', 'divide_', ['EXPR', 'EXPR'], 1.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Lambda expressions:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# EXPR -> apply_(FUNCTION, EXPR)
grammar.add_rule('EXPR', 'apply_', ['FUNCTION', 'EXPR'], 5.0)

# Here, 'lambda' is a special function that allows us to introduce a new bound
# variable (bv) of a certain type.
# The type is specified by bv_args. Here is how we might use it here:

# Creates a terminal of type bool -- e.g. y1
grammar.add_rule('FUNCTION', 'lambda', ['EXPR'], 1.0,  bv_type='BOOL', bv_args=None)

# BUt we can also use more complex situations, where the lambda is a thunk, or the bound variable
# is itself a function. These are commented out because FUNCTION is only expanded via application to
# a single EXPR, so the types these require are not supported by the above apply_

# Creates a thunk (lambda of no arguments)
#grammar.add_rule('FUNCTION', 'lambda', ['EXPR'], 1.0,  bv_type=None, bv_args=None)

# Creates a thunk lambda variable -- e.g y1()
#grammar.add_rule('FUNCTION', 'lambda', ['EXPR'], 1.0,  bv_type='BOOL', bv_args=[])

# Creates a lambda variable yi always called with an EXPR argument -- e.g., y1(plus(1,1))
#grammar.add_rule('FUNCTION', 'lambda', ['EXPR'], 1.0,  bv_type='BOOL', bv_args=['EXPR'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Conditional:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# if_ gets printed specially (see LOTlib.FunctionNode.__str__). Here COND is a name that is made up
# here for conditional expressions
grammar.add_rule('EXPR', 'if_', ['COND', 'EXPR', 'EXPR'], 1.0)
grammar.add_rule('COND', 'gt_', ['EXPR', 'EXPR'], 1.0)
grammar.add_rule('COND', 'eq_', ['EXPR', 'EXPR'], 1.0)

# Note that because if_ prints specially in FunctionNode, it is correctly handled (via short circuit evaluation)
# so that we don't eval both branches unnecessarily

if __name__ == "__main__":

    for _ in xrange(1000):

        t = grammar.generate() # Default is to generate from 'START'; else use 'START=t' to generate from type t

        # We can make this into a function by adding a lambda and a variable name, corresponding to
        # the argument "x" that we built into the grammar. This step is defaultly done by a a LOTHypothesis (see below)

        f = evaluate_expression('lambda x:%s'%t)

        print t # will call x.__str__ and display as a pythonesque string
        print map(f, range(0,10))

        # Alternatively, we can just make a LOTHypothesis, which is typically the only place in LOTlib we use trees
        from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
        h = LOTHypothesis(grammar, value=t, args=['x'])
        print map(h, range(0,10))


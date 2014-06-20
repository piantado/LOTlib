"""
    Routines for evaling 
"""

from LOTlib.Miscellaneous import raise_exception
from EvaluationException import EvaluationException


# All of these are defaulty in the context for eval. 
from LOTlib.Primitives.Arithmetic import *
from LOTlib.Primitives.Combinators import *
from LOTlib.Primitives.Features import *
from LOTlib.Primitives.Functional import *
from LOTlib.Primitives.Logic import *
from LOTlib.Primitives.Number import *
from LOTlib.Primitives.Semantics import *
from LOTlib.Primitives.SetTheory import *
from LOTlib.Primitives.Trees import *
from LOTlib.Primitives.Stochastics import *

import sys
def define_for_evaluator(name,function):
    """
        This function allows us to load new functions into the evaluation environment. 
        Defaultly all in LOTlib.Primitives are imported. However, we may want to add our
        own functions, and this makes that possible
        
        as in,
        
        define_for_evaluator('flatten', flatten)
        
        where flatten is a function that is defined in the calling context
        
        TODO: Add more convenient means for importing more methods
    """
    sys.modules['__builtin__'].__dict__[name] = function 


"""
The Y combinator
#example:
#fac = lambda f: lambda n: (1 if n<2 else n*(f(n-1)))
#Y(fac)(10)
"""
Y = lambda f: (lambda x: x(x)) (lambda y : f(lambda *args: y(y)(*args)) )


MAX_RECURSION = 25
def Y_bounded(f):
    """
    A fancy fixed point iterator that only goes MAX_RECURSION deep, else throwing a a RecusionDepthException
    """
    return (lambda x, n: x(x, n)) (lambda y, n: f(lambda *args: y(y, n+1)(*args)) if n < MAX_RECURSION else raise_exception(EvaluationException()), 0)




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
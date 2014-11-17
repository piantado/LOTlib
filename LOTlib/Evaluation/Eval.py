"""
    Routines for evaling
"""
import sys

from LOTlib.Miscellaneous import raise_exception
from EvaluationException import EvaluationException

# All of these are defaulty in the context for eval.
from Primitives.Arithmetic import *
from Primitives.Combinators import *
from Primitives.Features import *
from Primitives.Functional import *
from Primitives.Logic import *
from Primitives.Number import *
from Primitives.Semantics import *
from Primitives.SetTheory import *
from Primitives.Trees import *
from Primitives.Stochastics import *

def register_primitive(function, name=None):
    """
        This allows us to load new functions into the evaluation environment.
        Defaultly all in LOTlib.Primitives are imported. However, we may want to add our
        own functions, and this makes that possible. As in,

        register_primitive(flatten)

        or

        register_primitive(flatten, name="myflatten")

        where flatten is a function that is defined in the calling context and name
        specifies that it takes a different name when evaled in LOTlib

        TODO: Add more convenient means for importing more methods
    """

    if name is None: # if we don't specify a name
        name = function.__name__

    sys.modules['__builtin__'].__dict__[name] = function

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~ Evaluation of expressions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evaluate_expression(e):
    """
    Evaluate the expression, wrapping in an error in case it can't be evaled
    """
    assert isinstance(e, str)
    try:
        return eval(e)
    except Exception as ex:
        print "*** Error in evaluate_expression:", ex
        raise ex

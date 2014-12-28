from LOTlib.Evaluation.Eval import LOTlib_primitive

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Combinators -- all curried
# NOTE: Evaluation.CombinatoryLogic also uses combinators, but
#       implements a direct evaluator rather than these lambdas
#       which will get evaled in python
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@LOTlib_primitive
def I_(x):
    """I combinator - identity function.

    Example:
        a = I_('hello')
        a == 'hello'

    """
    return x

@LOTlib_primitive
def K_(x):
    """K combinator - one-argument constant function.

    Example:
        y = K_('hello')
        y(anything) == 'hello'

    """
    return lambda y: x

@LOTlib_primitive
def S_(x):
    """Substitution operator - magic stuff.

    (S x y z) = (x z (y z))
    (S x) --> lambda y lambda z:

    If we wrap y in a K_ function, it will ignores z and really we have x(z,y).

    """
    return lambda y: lambda z: x(z)(y(z))

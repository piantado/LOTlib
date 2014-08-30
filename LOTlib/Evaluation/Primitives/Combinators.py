from Primitives import LOTlib_primitive

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Combinators -- all curried
# NOTE: Evaluation.CombinatoryLogic also uses combinators, but
#       implements a direct evaluator rather than these lambdas
#       which will get evaled in python
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@LOTlib_primitive
def I_(x):
    return x

@LOTlib_primitive
def K_(x): # constant function
    return (lambda y: x)

@LOTlib_primitive
def S_(x): #(S x y z) = (x z (y z))
    # (S x) --> lambda y lambda z:
    return lambda y: lambda z: x(z)( y(z) )

from Primitives import LOTlib_primitive

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Access arbitrary features
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Some of our own primitivesS
@LOTlib_primitive
def is_color_(x,y): return (x.color == y)

@LOTlib_primitive
def is_shape_(x,y): return (x.shape == y)

@LOTlib_primitive
def is_pattern_(x,y): return (x.pattern == y)

@LOTlib_primitive
def switch_(i,*ar):
    """
        Index into an array. NOTE: with run-time priors, the *entire* array gets evaluated.
        If you want to avoid this, use switchf_, which requires lambdas
    """
    return ar[i]

@LOTlib_primitive
def switchf_(i,x,*ar):
    return ar[i](x)
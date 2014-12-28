from LOTlib.Evaluation.Eval import LOTlib_primitive
from LOTlib.Miscellaneous import raise_exception
from LOTlib.Evaluation.EvaluationException import RecursionDepthException

## TODO: Add transitive closure of an operation


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~ The Y combinator and a bounded version
# example:
# fac = lambda f: lambda n: (1 if n<2 else n*(f(n-1)))
# Y(fac)(10)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Y = lambda f: (lambda x: x(x))(lambda y: f(lambda *args: y(y)(*args)) )
MAX_RECURSION = 25


def Y_bounded(f):
    """
    A fancy fixed point iterator that only goes MAX_RECURSION deep, else throwing a a RecusionDepthException
    """
    return (lambda x, n: x(x, n))(lambda y, n: f(lambda *args: y(y, n+1)(*args))
                                  if n < MAX_RECURSION else raise_exception(RecursionDepthException), 0)


def Ystar(*l):
    """
    The Y* combinator for mutually recursive functions. Holy shit.

    (define (Y* . l)
          ((lambda (u) (u u))
            (lambda (p) (map (lambda (li) (lambda x (apply (apply li (p p)) x))) l))))

    See:
    http://okmij.org/ftp/Computation/fixed-point-combinators.html]
    http://stackoverflow.com/questions/4899113/fixed-point-combinator-for-mutually-recursive-functions

    E.g., here is even/odd:

    even,odd = Ystar( lambda e,o: lambda x: (x==0) or o(x-1), \
                          lambda e,o: lambda x: (not x==0) and e(x-1) )

        Note that we require a lambda e,o on the outside so that these can have names inside.
    """

    return (lambda u: u(u))(lambda p: map(lambda li: lambda x: apply(li, p(p))(x), l))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Lambda calculus & Scheme
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@LOTlib_primitive
def lambda_(f,args):
    f.args = args
    return f

@LOTlib_primitive
def map_(f,A):
    return [f(a) for a in A]

@LOTlib_primitive
def apply_(f,*args):
    return f(*args)

@LOTlib_primitive
def cons_(x,y):
    return [x,y]

@LOTlib_primitive
def cdr_(x):
    try:    return x[1:]
    except IndexError: return []

rest_  = cdr_

@LOTlib_primitive
def car_(x):
    try:    return x[0]
    except IndexError: return []

first_ = car_

@LOTlib_primitive
def filter_(f,x):
    return filter(f,x)

@LOTlib_primitive
def filterset_(f,x):
    return set(filter(f,x))

@LOTlib_primitive
def mapset_(f,A):
    return {f(a) for a in A}

@LOTlib_primitive
def Ystar_(*args):
    return Ystar(*args)



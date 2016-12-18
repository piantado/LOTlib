"""
Here is a version where the string operations manipulate dictionaries mapping strings to probabilities.
This prevents us from having to simulate everything. NOTE: It requires RecursionDepthException to
be handled as generating empty string

TODO: Add BOOL -- and or not, equality, etc.

"""

from LOTlib.Grammar import Grammar

base_grammar = Grammar()
base_grammar.add_rule('START', '', ['LIST'], 1.0) # stochasic computation
base_grammar.add_rule('START', 'detfunc(lambda rec: lambda x: %s, %s)', ['DETLIST', 'LIST'], 1.0) # deterministic computation (map)

base_grammar.add_rule('LIST', 'if_d', ['BOOL', 'LIST', 'LIST'], 1.)

base_grammar.add_rule('LIST', 'cons_d', ['ATOM', 'LIST'], 1./6.)
base_grammar.add_rule('LIST', 'cons_d', ['LIST', 'LIST'], 1./6.)
base_grammar.add_rule('LIST', 'cdr_d', ['LIST'], 1./3.)
base_grammar.add_rule('LIST', 'car_d', ['LIST'], 1./3.)

base_grammar.add_rule('LIST', '', ['ATOM'], 3.0)
base_grammar.add_rule('LIST', '{\'\':0.0}', None, 1.0)

base_grammar.add_rule('BOOL', 'empty_d', ['LIST'], 1.)
base_grammar.add_rule('BOOL', 'flip_d(p=%s)', ['PROB'], 1.)

for i in xrange(1,10):
    base_grammar.add_rule('PROB', '0.%s' % i, None, 1.)

# can call memoized and non-memoized versions of myself
base_grammar.add_rule('LIST', 'recurse_(lex_)',   None, 5.0) # can call myself, but I have to pass in the other args (recurse is already given to itself)
base_grammar.add_rule('LIST', 'lex_(%s)',   ['SELFF'], 1.0/2.0) # can call other words, but I have to say whether to memoize or not
base_grammar.add_rule('SELFF', '-1', None, 0.00001)  #needed to tha tif lex is called on the first word, the grammar still works. It has low prob so its not generated wonce the other lex are added

# apply a deterministic function to a stochastic set. Here, x and rec are the arguments that detfunc manages
# we don't use built-in lambdas here because DETLIST is different (deterministic) functions and we need to pass x and rec
# and detfunc has to also implement its own Y-combinator
base_grammar.add_rule('DETLIST', '(%s if %s else %s)', ['DETLIST', 'DETBOOL', 'DETLIST'], 1.)

base_grammar.add_rule('DETLIST', 'cons_', ['DETATOM', 'DETLIST'], 1./6.)
base_grammar.add_rule('DETLIST', 'cons_', ['DETLIST', 'DETLIST'], 1./6.)
base_grammar.add_rule('DETLIST', 'cdr_', ['DETLIST'], 1./3.)
base_grammar.add_rule('DETLIST', 'car_', ['DETLIST'], 1./3.)

base_grammar.add_rule('DETLIST', 'x', None, 10.0)

base_grammar.add_rule('DETLIST', '', ['DETATOM'], 3.0)
base_grammar.add_rule('DETLIST', "''", None, 1.0)

base_grammar.add_rule('DETBOOL', 'empty_', ['DETLIST'], 1.)

from LOTlib.Primitives import primitive
from LOTlib.Miscellaneous import Infinity, logplusexp, lambdaMinusInfinity, flatten2str
from collections import defaultdict

Y = lambda f: (lambda x: x(x))(lambda y: f(lambda *args: y(y)(*args))) #https://rosettacode.org/wiki/Y_combinator#Python

@primitive
def detfunc(f, l):
    """ Apply a deterministic function to a stochastic mapping (dict from values to log probs).
        This works item-wise """
    out = defaultdict(lambdaMinusInfinity)
    for x,lp in l.items():
        v = flatten2str(Y(f)(x))
        out[ v ] = logplusexp(out[v], lp)
    return out





if __name__ == "__main__":
    from Model import InnerHypothesis

    for _ in xrange(1000):
        h = InnerHypothesis(grammar=base_grammar)
        print h
        print h(h)

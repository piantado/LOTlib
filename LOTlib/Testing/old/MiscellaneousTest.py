"""
        class to test Miscellaneous.py
        follows the standards in https://docs.python.org/2/library/unittest.html
"""


import unittest

from LOTlib.Miscellaneous import *


class MiscellaneousTest(unittest.TestCase):



    # initialization that happens before each test is carried out
    def setUp(self):
        self.array = [10,9,8,7,6,5,4,3,2,1]

    # tests the first, second, ... functions
    def test_number_functions(self):
        self.assertEqual(first(self.array), 10)
        self.assertEqual(second(self.array), 9)
        self.assertEqual(third(self.array), 8)
        self.assertEqual(fourth(self.array), 7)
        self.assertEqual(fifth(self.array), 6)
        self.assertEqual(sixth(self.array), 5)
        self.assertEqual(seventh(self.array), 4)
        self.assertEqual(eighth(self.array), 3)
        self.assertTrue(isinstance(dropfirst(self.array), types.GeneratorType))

    def test_None2Empty(self):
        none = None
        # Treat Nones as empty
        self.assertEqual(None2Empty(none), [])
        self.assertEqual(None2Empty(self.array), self.array)

    # def test_make_mutable(self):
    #       # TODO: update with other types
    #       if isinstance(x, frozenset): return set(x)
    #       elif isinstance(x, tuple): return list(x)
    #       else: return x

    # def test_make_immutable(self):
    #       # TODO: update with other types
    #       if isinstance(x, set ): return frozenset(x)
    #       elif isinstance(x, list): return tuple(x)
    #       else: return x

    def test_unlist_singleton(self):
        """
                Remove any sequences of nested lists with one element.

                e.g. [[[1,3,4]]] -> [1,3,4]
        """
        self.assertEqual(unlist_singleton(self.array), self.array)
        self.assertEqual(unlist_singleton([[[1,3,4]]]), [1,3,4])
        self.assertEqual(unlist_singleton([]), [])

    def test_list2sexpstr(self):
        """
                Prints a python list-of-lists as an s-expression

                [['K', 'K'], [['S', 'K'], ['I', 'I']]] --> ((K K) ((S K) (I I)))
        """
        self.assertEqual(list2sexpstr([['K', 'K'], [['S', 'K'], ['I', 'I']]]), '((K K) ((S K) (I I)))')
        self.assertEqual(list2sexpstr([]), '()')
        # s = re.sub(r'[\'\",]', r'', str(lst))
        # s = re.sub(r'\[', '(', s) # changed r'(' to '('
        # s = re.sub(r'\]', ')', s) # changed r')' to ')'
        # return s


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display functions
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def test_q(self):
        """
                Quotes a string
        """
        self.assertEqual(q("This is a string"), "'This is a string'")
        self.assertEqual(q("This is a string", quote="\""), "\"This is a string\"")
        self.assertEqual(q([]), "'[]'")
        self.assertTrue(qq("This is a string") == q("This is a string", quote='"') == "\"This is a string\"")

    # def test_qq(x): return q(x,quote="\"")

    # def test_display(x): print x

    # # for functional programming, print something and return it
    # def test_printr(x):
    #       print x
    #       return x

    # def test_r2(x): return round(x,2)
    # def test_r3(x): return round(x,3)
    # def test_r4(x): return round(x,4)
    # def test_r5(x): return round(x,5)

    # def test_tf201(x):
    #       if x: return 1
    #       else: return 0


    # ## Functions for I/O
    # def test_display_option_summary(obj):
    #       """
    #               Prints out a friendly format of all options -- for headers of output files
    #               This takes in an OptionParser object as an argument. As in, (options, args) = parser.parse_args()
    #       """
    #       from time import strftime, time, localtime
    #       import os

    #       print "####################################################################################################"
    #       try: print "# Username: ", os.getlogin()
    #       except OSError: pass

    #       try: print "# Date: ", strftime("%Y %b %d (%a) %H:%M:%S", localtime(time()) )
    #       except OSError: pass

    #       try: print "# Uname: ", os.uname()
    #       except OSError: pass

    #       try: print "# Pid: ", os.getpid()
    #       except OSError: pass

    #       for slot in dir(obj):
    #               attr = getattr(obj, slot)
    #               if not isinstance(attr, (types.BuiltinFunctionType, types.FunctionType, types.MethodType)) and (slot is not "__doc__") and (slot is not "__module__"):
    #                       print "#", slot, "=", attr
    #       print "####################################################################################################"



    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # Genuine Miscellany
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # # a wrapper so we can call this in the below weirdo composition
    # def test_raise_exception(e): raise e

    # def test_ifelse(x,y,z):
    #       if x: return y
    #       else: return z

    # def test_unique(gen):
    #       """
    #               Make a generator unique, returning each element only once
    #       """
    #       s = set()
    #       for gi in gen:
    #               if gi not in s:
    #                       yield gi
    #                       s.add(gi)

    # def test_UniquifyFunction(gen):
    #       """
    #               A decorator to make a function only return unique values
    #       """
    #       def test_f(*args, **kwargs):
    #               for x in unique(gen(*args, **kwargs)):
    #                       yield x
    #       return f

    # def test_flatten(expr):
    #       """
    #               Flatten lists of lists, via stackoverflow
    #       """
    #       def test_flatten_(expr):
    #               #print 'expr =', expr
    #               if expr is None or not isinstance(expr, collections.Iterable) or isinstance(expr, str):
    #                       yield expr
    #               else:
    #                       for node in expr:
    #                               #print node, type(node)
    #                               if (node is not None) and isinstance(node, collections.Iterable) and (not isinstance(node, str)):
    #                                       #print 'recursing on', node
    #                                       for sub_expr in flatten_(node):
    #                                               yield sub_expr
    #                               else:
    #                                       #print 'yielding', node
    #                                       yield node

    #       return tuple([x for x in flatten_(expr)])

    # def test_flatten2str(expr, sep=' '):
    #       try:
    #               if expr is None: return ''
    #               else:            return sep.join(flatten(expr))
    #       except TypeError:
    #               print "Error in flatter2str:", expr
    #               raise TypeError

    # def test_weave(*iterables):
    #       """
    #       Intersperse several iterables, until all are exhausted.
    #       This nicely will weave together multiple chains

    #       from: http://www.ibm.com/developerworks/linux/library/l-cpyiter/index.html
    #       """

    #       iterables = map(iter, iterables)
    #       while iterables:
    #               for i, it in enumerate(iterables):
    #                       try: yield it.next()
    #                       except StopIteration: del iterables[i]

    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # Math functions
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # # Special handilng for numpypy that doesn't use gammaln, assertion error otherwise
    # try:
    #       from scipy.special import gammaln
    # except ImportError:
    #       # Die if we try to use this in numpypy
    #       def test_gammaln(*args, **kwargs): assert False

    # ## This is just a wrapper to avoid logsumexp([-inf, -inf, -inf...]) warnings
    # try:
    #       from scipy.misc import logsumexp as scipy_logsumexp
    # except ImportError:
    #       try:
    #               from scipy.maxentropy import logsumexp as scipy_logsumexp
    #       except ImportError:
    #               # fine, our own version, no numpy
    #               def test_scipy_logsumexp(v):
    #                       m = max(v)
    #                       return m+log(sum(map( lambda x: exp(x-m), v)))

    # def test_logsumexp(v):
    #       """
    #               Logsumexp - our own version wraps the scipy to handle -infs
    #       """
    #       if max(v) > -Infinity: return scipy_logsumexp(v)
    #       else: return -Infinity

    # def test_lognormalize(v):
    #       return v - logsumexp(v)

    # def test_logplusexp(a, b):
    #       """
    #               Two argument version. No cast to numpy, so faster
    #       """
    #       m = max(a,b)
    #       return m+log(exp(a-m)+exp(b-m))

    # def test_beta(a):
    #       """ Here a is a vector (of ints or floats) and this computes the Beta normalizing function,"""
    #       return np.sum(gammaln(np.array(a, dtype=float))) - gammaln(float(sum(a)))


    # def test_normlogpdf(x, mu, sigma):
    #       """ The log pdf of a normal distribution """
    #       #print x, mu
    #       return math.log(math.sqrt(2. * pi) * sigma) - ((x - mu) * (x - mu)) / (2.0 * sigma * sigma)

    # def test_norm_lpdf_multivariate(x, mu, sigma):
    #       # Via http://stackoverflow.com/questions/11615664/multivariate-normal-density-in-python
    #       size = len(x)

    #       # some checks:
    #       if size != len(mu) or (size, size) != sigma.shape: raise NameError("The dimensions of the input don't match")
    #       det = np.linalg.det(sigma)
    #       if det == 0: raise NameError("The covariance matrix can't be singular")

    #       norm_const = - size*log(2.0*pi)/2.0 - log(det)/2.0
    #       #norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
    #       x_mu = np.matrix(x - mu)
    #       inv = np.linalg.inv(sigma)
    #       result = -0.5 * (x_mu * inv * x_mu.T)
    #       return norm_const + result

    # def test_logrange(mn,mx,steps):
    #       """
    #               Logarithmically-spaced steps from mn to mx, with steps number inbetween
    #               mn - min value
    #               mx - max value
    #               steps - number of steps between. When 1, only mx is returned
    #       """
    #       mn = np.log(mn)
    #       mx = np.log(mx)
    #       r = np.arange(mn, mx, (mx-mn)/(steps-1))
    #       r = np.append(r, mx)
    #       return np.exp(r)

    # def test_geometric_ldensity(n,p):
    #       """ Log density of a geomtric distribution """
    #       return (n-1)*log(1.0-p)+log(p)

    # from math import expm1, log1p
    # def test_log1mexp(a):
    #       """
    #               Computes log(1-exp(a)) according to Machler, "Accurately computing ..."
    #               Note: a should be a large negative value!
    #       """
    #       if a > 0: print >>sys.stderr, "# Warning, log1mexp with a=", a, " > 0"
    #       if a < -log(2.0): return log1p(-exp(a))
    #       else:             return log(-expm1(a))


    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # Sampling functions
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # def test_sample1(*args): return sample_one(*args)

    # def test_sample_one(*args):
    #       if len(args) == 1: return sample(args[0],1)[0] # use the list you were given
    #       else:              return sample(args, 1)[0]   # treat the arguments as a list

    # def test_flip(p): return (random() < p)


    # ## TODO: THIS FUNCTION SUCKS PLEASE FIX IT
    # ## TODO: Change this so that if N is large enough, you sort
    # # takes unnormalized probabilities and returns a list of the log probability and the object
    # # returnlist makes the return always a list (even if N=1); otherwise it is a list for N>1 only
    # # NOTE: This now can take probs as a function, which is then mapped!
    # def test_weighted_sample(objs, N=1, probs=None, log=False, return_probability=False, returnlist=False, Z=None):
    #       """
    #               When we return_probability, it is *always* a log probability
    #       """
    #       # check how probabilities are specified
    #       # either as an argument, or attribute of objs (either probability or lp
    #       # NOTE: THis ALWAYS returns a log probability

    #       if len(objs) == 0: return None

    #       # convert to support indexing if we need it
    #       if isinstance(objs, set):
    #               objs = list(objs)

    #       myprobs = None
    #       if probs is None: # defatest_ultly, we use .lp
    #               myprobs = [1.0] * len(objs) # sample uniform
    #       elif isinstance(probs, types.FunctionType): # NOTE: this does not work for class instance methods
    #               myprobs = map(probs, objs)
    #       else:
    #               myprobs = map(float, probs)

    #       # Now normalize and run
    #       if Z is None:
    #               if log: Z = logsumexp(myprobs)
    #               else: Z = sum(myprobs)
    #       #print log, myprobs, Z
    #       out = []

    #       for n in range(N):
    #               r = random()
    #               for i in range(len(objs)):
    #                       if log: r = r - exp(myprobs[i] - Z) # log domain
    #                       else: r = r - (myprobs[i]/Z) # probability domain
    #                       #print r, myprobs
    #                       if r <= 0:
    #                               if return_probability:
    #                                       lp = 0
    #                                       if log: lp = myprobs[i] - Z
    #                                       else:   lp = math.log(myprobs[i]) - math.log(Z)

    #                                       out.append( [objs[i],lp] )
    #                                       break
    #                               else:
    #                                       out.append( objs[i] )
    #                                       break

    #       if N == 1 and (not returnlist): return out[0]  # don't give back a list if you just want one
    #       else:      return out

    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # Lambda calculus
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # # Some innate lambdas
    # def test_lambdaZero(*x): return 0
    # def test_lambdaOne(*x): return 1
    # def test_lambdaNull(*x): return []
    # def test_lambdaNone(*x): return None
    # def test_lambdaTrue(*x): return True
    # def test_lambdaFalse(*x): return True
    # def test_lambdaNAN(*x): return float("nan")

    # def test_lambda_str(fn):
    #       """
    #               A nicer printer for pure lambda calculus
    #       """
    #       if fn is None: # just pass these through -- simplifies a lot
    #               return None
    #       elif fn.name == '':
    #               assert len(fn.args)==1
    #               return lambda_str(fn.args[0])
    #       elif fn.name == 'lambda':
    #               assert len(fn.args)==1
    #               #return u"\u03BB%s.%s" % (fn.bv_name, lambda_str(fn.args[0]))
    #               return "L%s.%s" % (fn.bv_name, lambda_str(fn.args[0]))
    #       elif fn.name == 'apply_':
    #               assert len(fn.args)==2
    #               if fn.args[0].name == 'lambda':
    #                       return "((%s)(%s))" % tuple(map(lambda_str, fn.args))
    #               else:
    #                       return "(%s(%s))" % tuple(map(lambda_str, fn.args))
    #       else:
    #               assert fn.args is None
    #               return str(fn.name)


# A Test Suite composed of all tests in this class
def suite():
    return unittest.TestLoader().loadTestsFromTestCase(MiscellaneousTest)













# main code to run the test
if __name__ == '__main__':
    unittest.main()

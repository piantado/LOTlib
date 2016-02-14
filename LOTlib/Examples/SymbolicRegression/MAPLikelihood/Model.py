"""
    Symbolic regression with MAP estimation of parameters
    For use with Demo run via
        python Demo.py --model=SymbolicRegression.MAPLikelihood --alsoprint='lambda h: h.parameters'

"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Examples.SymbolicRegression import make_data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We use the same grammar, but add constant terms
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Examples.SymbolicRegression import grammar, TERMINAL_WEIGHT

NCONSTANTS = 4  # How many?
CONSTANT_NAMES = ['C%i'%i for i in xrange(NCONSTANTS) ]

# Supplement the grammar with constant names
for c in CONSTANT_NAMES:
    grammar.add_rule('EXPR', c, None, TERMINAL_WEIGHT)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define a custom hypothesis
# This must fit the constants when we eval
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy
from scipy.optimize import fmin
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Likelihoods.GaussianLikelihood import GaussianLikelihood
from LOTlib.Miscellaneous import normlogpdf, Infinity, attrmem

MAXITER=100 # max iterations for the optimization to run
MAX_INITIALIZE=25 # max number of random numbers to try initializing with

class MAPSymbolicRegressionHypothesis(GaussianLikelihood, LOTHypothesis):
    """
    This is a quick hack to try out symbolic regression with constants just fit.
    This hacks it by defining a self.parameters that are automatically read from
    get_function_responses (overwritten). We can then change them and repeatedly compute the
    likelihood to optimize
    """

    def __init__(self, constant_sd=1.0, fit_only_once=True, **kwargs):
        """
        :param constant_sd: The SD of our constants in the prior
        :param fit_only_once: Do we fit multiple times or just take the first?
        """
        LOTHypothesis.__init__(self, grammar, display='lambda x,'+','.join(CONSTANT_NAMES)+": %s", **kwargs)

        self.constant_sd=constant_sd # also the prior SD
        self.parameters = self.sample_constants()
        self.fit_only_once = fit_only_once

    def __call__(self, *vals):
        """
            Must overwrite call in order to include the constants
        """
        vals = list(vals)
        vals.extend(self.parameters)
        return LOTHypothesis.__call__(self, *vals)

    def sample_constants(self):
        """ Return a random sample of the constants (does NOT set them) """
        return numpy.random.normal(0.0, self.constant_sd, len(CONSTANT_NAMES))

    @attrmem('prior')
    def compute_prior(self):
        # Add together the LOT prior and the constant prior, here just a gaussian
        return LOTHypothesis.compute_prior(self) +\
               sum(map(lambda x: normlogpdf(x,0.0,self.constant_sd), self.parameters))

    @attrmem('likelihood')
    def compute_likelihood(self, data, shortcut=False):
        """ A pseudo-likelihood corresponding to that under the best fitting params """

        def to_maximize(fit_params):
            self.parameters = fit_params.tolist() # set these
            # And return the original likelihood, which by get_function_responses above uses this
            constant_prior = sum(map(lambda x: normlogpdf(x,0.0,self.constant_sd), self.parameters))
            return -(LOTHypothesis.compute_likelihood(self, data) + constant_prior)

        bestval, bestparms = -Infinity, numpy.zeros(NCONSTANTS)
        for init in xrange(MAX_INITIALIZE):
            # pick a random starting position
            params = self.sample_constants()

            # minimize
            res = fmin(to_maximize, params, disp=False, maxiter=MAXITER)
            resval = -to_maximize(res) # negative makes it log prob again

            if resval > bestval:
                bestval, bestparms = resval, params

            # We may want to stop
            if self.fit_only_once and bestval < Infinity:
                break

        self.parameters = bestparms

        return bestval

def make_hypothesis(**kwargs):
    return MAPSymbolicRegressionHypothesis(**kwargs)
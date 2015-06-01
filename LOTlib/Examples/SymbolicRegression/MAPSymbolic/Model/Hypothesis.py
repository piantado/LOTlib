
import numpy
from numpy.random import normal
from scipy.optimize import fmin
from LOTlib.Evaluation.Eval import evaluate_expression
from LOTlib.Hypotheses.GaussianLOTHypothesis import GaussianLOTHypothesis
from LOTlib.Miscellaneous import normlogpdf, Infinity, attrmem
from Grammar import grammar, CONSTANT_NAMES, NCONSTANTS


MAXITER=100 # max iterations for the optimization to run
MAX_INITIALIZE=25 # max number of random numbers to try initializing with

class MAPSymbolicRegressionHypothesis(GaussianLOTHypothesis):
    """
    This is a quick hack to try out symbolic regression with constants just fit.
    This hacks it by defining a self.CONSTANT_VALUES that are automatically read from
    get_function_responses (overwritten). We can then change them and repeatedly compute the
    likelihood to optimize
    """

    def __init__(self, constant_sd=1.0, **kwargs):
        GaussianLOTHypothesis.__init__(self, grammar, args=['x']+CONSTANT_NAMES, **kwargs)

        self.CONSTANT_VALUES = numpy.zeros(NCONSTANTS)
        self.constant_sd=constant_sd

    def compile_function(self):
        """
        Overwrite this from FunctionHypothesis. Here, we add args for the constants so we can use them
        """
        return evaluate_expression(str(self))

    def __call__(self, *vals):
        """
            Must overwrite call in order to include the constants
        """
        vals = list(vals)
        vals.extend(self.CONSTANT_VALUES)
        return GaussianLOTHypothesis.__call__(self, *vals)

    @attrmem('prior')
    def compute_prior(self):
        # Add together the structural prior and the constant prior
        return GaussianLOTHypothesis.compute_prior(self) +\
               sum(map(lambda x: normlogpdf(x,0.0,self.constant_sd), self.CONSTANT_VALUES))

    @attrmem('likelihood')
    def compute_likelihood(self, data):

        def to_maximize(fit_params):
            self.CONSTANT_VALUES = fit_params.tolist() # set these
            # And return the original likelihood, which by get_function_responses above uses this
            constant_prior = sum(map(lambda x: normlogpdf(x,0.0,self.constant_sd), self.CONSTANT_VALUES))
            return -(GaussianLOTHypothesis.compute_likelihood(self, data) + constant_prior)

        for init in xrange(MAX_INITIALIZE):
            p0 = normal(0.0, self.constant_sd, len(CONSTANT_NAMES))
            res = fmin(to_maximize, p0, disp=False, maxiter=MAXITER)
            if to_maximize(res) < Infinity: break

        maxval = to_maximize(res)
        self.CONSTANT_VALUES = res

        if maxval < Infinity:
            return -maxval ## must invert since it's a negative
        else:
            return -Infinity

def make_hypothesis():
    return MAPSymbolicRegressionHypothesis()
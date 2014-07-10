from numpy.random import normal
from scipy.optimize import fmin

from LOTlib.Hypotheses.GaussianLOTHypothesis import GaussianLOTHypothesis
from LOTlib.Miscellaneous import evaluate_expression, normlogpdf, Infinity

from Grammar import CONSTANT_NAMES

MAXITER=100 # max iterations for the optimization to run
MAX_INITIALIZE=25 # max number of random numbers to try initializing with

CONSTANT_SD = 1.0 # TODO: Put this as an attribute

class MAPSymbolicRegressionHypothesis(GaussianLOTHypothesis):
    """
        This is a quick hack to try out symbolic regression with constants just fit. 
        This hacks it by defining a self.CONSTANT_VALUES that are automatically read from 
        get_function_responses (overwritten). We can then change them and repeatedly compute the 
        likelihood to optimize
    """
    
    def value2function(self, value):
        """
        Overwrite this from FunctionHypothesis. Here, we add args for the constants so we can use them
        """
        #print ">>", self.args, CONSTANT_NAMES
        return evaluate_expression(value, args=self.args+CONSTANT_NAMES)
        
    def __call__(self, *vals):
        """
            Must overwrite call in order to include the constants
        """
        vals = list(vals)
        vals.extend(self.CONSTANT_VALUES)
        return GaussianLOTHypothesis.__call__(self, *vals)
    
    def compute_prior(self):
        self.prior = GaussianLOTHypothesis.compute_prior(self)
        self.prior += sum(map(lambda x: normlogpdf(x,0.0,CONSTANT_SD), self.CONSTANT_VALUES))
        self.posterior_score = self.prior + self.likelihood
        return self.prior
    
    def compute_likelihood(self, data):
        
        def toMaximize(fit_params):
            self.CONSTANT_VALUES = fit_params.tolist() # set these
            # And return the original likelihood, which by get_function_responses above uses this
            constant_prior = sum(map(lambda x: normlogpdf(x,0.0,CONSTANT_SD), self.CONSTANT_VALUES))
            return -(GaussianLOTHypothesis.compute_likelihood(self, data) + constant_prior)

        for init in xrange(MAX_INITIALIZE):
            p0 = normal(0.0, CONSTANT_SD, len(CONSTANT_NAMES))
            res = fmin(toMaximize, p0, disp=False, maxiter=MAXITER)
            if toMaximize(res) < Infinity: break
        
        
        maxval = toMaximize(res)
        self.CONSTANT_VALUES = res
        
        if maxval < Infinity:  self.likelihood = -maxval # must invert since it's a negative
        else:                  self.likelihood = -Infinity
            
        self.posterior_score = self.prior + self.likelihood
        return self.likelihood
